import io
import logging
import cv2
import numpy as np
from typing import Any

from worker.utils.download import download_models
from worker.utils.constants import NDArray
from worker.utils.image_preprocessing import color_adjust
from worker.segmentation.inference import generate_segmentation_preds
from worker.utils.image_postprocessing import (
    filter_segmentation_preds,
    make_lines_stronger,
)
from worker.utils.bounding_box import (
    create_bounding_ellipses,
    create_rotated_bboxes,
    SymbolBoundingBoxes,
)
from worker.detection.staff import (
    break_wide_fragments,
    find_staff_anchors,
    predict_other_anchors_from_clefs,
    filter_unusual_anchors,
    find_raw_staffs_by_connecting_fragments,
    remove_duplicate_staffs,
    resample_staffs,
    filter_edge_of_vision,
)
from worker.detection.note import combine_noteheads_with_stems, add_notes_to_staffs
from worker.detection.bar_line import detect_bar_lines, add_bar_lines_to_staffs
from worker.detection.rest import add_rests_to_staffs
from worker.detection.brace_dot import (
    prepare_brace_dot_image,
    find_braces_brackets_and_grand_staff_lines,
)
from worker.detection.accidental import add_accidentals_to_staffs
from worker.parser.staff import parse_staffs, merge_staffs_across_pages
from worker.musicxml.accidental_rules import maintain_accidentals
from worker.musicxml.xml_generator import generate_xml
from worker.utils.debug import write_debug_image
from worker.classes.results import Page

logging.basicConfig(
    level=logging.INFO,
    format="(%(name)s:%(lineno)s) - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_inference(image_bytes: bytes, is_first_page: bool) -> dict[str, Any]:
    # download models
    download_models()

    # DETECT STAFFS IN IMAGE
    logger.info("Detecting staffs")
    ## LOADING/PREPROCESSING SEGMENTATION PREDICTIONS
    logger.info("Loading segmentation")
    ### IMAGE PREPROCESSING
    # image = cv2.imread(image_path)
    # image = autocrop(image)
    # image = resize_image(image)

    # image = preprocess_image_from_bytes(image_bytes)

    image: NDArray = np.load(io.BytesIO(image_bytes))
    preprocessed, _ = color_adjust(image)

    ### MODEL INFERENCE
    predictions = generate_segmentation_preds(image, preprocessed)

    ### IMAGE POSTPROCESSING
    predictions = filter_segmentation_preds(predictions)
    predictions.staff = make_lines_stronger(predictions.staff)
    logger.info("Loaded segmentation")

    write_debug_image(image, "1_staff.png", binary_map=predictions.staff)
    write_debug_image(image, "2_symbols.png", binary_map=predictions.symbols)
    write_debug_image(image, "3_stems_rests.png", binary_map=predictions.stems_rests)
    write_debug_image(image, "4_notehead.png", binary_map=predictions.notehead)
    write_debug_image(image, "5_clefs_keys.png", binary_map=predictions.clefs_keys)

    ## PREDICTING SYMBOLS
    logger.info("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead)
    logger.info("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bboxes(
        predictions.staff,
        skip_merging=True,
        min_size=(5, 1),
        max_size=(1000 * 10, 100),
    )
    logger.info("Creating bounds for clefs_keys")
    clefs_keys = create_rotated_bboxes(
        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
    )
    logger.info("Creating bounds for accidentals")
    accidentals = create_rotated_bboxes(
        predictions.clefs_keys, min_size=(5, 5), max_size=(100, 100)
    )
    logger.info("Creating bounds for stems_rests")
    stems_rests = create_rotated_bboxes(predictions.stems_rests)
    logger.info("Creating bounds for bar_lines")
    kernel = np.ones((5, 3), np.uint8)
    bar_line_img = cv2.dilate(predictions.stems_rests, kernel, iterations=1)
    bar_lines = create_rotated_bboxes(bar_line_img, skip_merging=True, min_size=(1, 5))
    symbols = SymbolBoundingBoxes(
        noteheads, staff_fragments, clefs_keys, accidentals, stems_rests, bar_lines
    )
    logger.info("Predicted symbols")

    write_debug_image(image, "6_ellipses.png", drawables=noteheads)
    write_debug_image(image, "7_staff_fragments.png", drawables=staff_fragments)
    write_debug_image(image, "8_clefs_keys.png", drawables=clefs_keys)
    write_debug_image(image, "9_accidentals.png", drawables=accidentals)
    write_debug_image(image, "10_stems_rests.png", drawables=stems_rests)
    write_debug_image(image, "11_bar_line_img.png", binary_map=bar_line_img)
    write_debug_image(image, "12_bar_lines.png", drawables=bar_lines)

    ## BREAKING WIDE FRAGMENTS
    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    logger.info(f"Found {len(symbols.staff_fragments)} staff line fragments")

    write_debug_image(
        image, "13_staff_fragments.png", drawables=symbols.staff_fragments
    )

    ## COMBINING NOTEHEADS WITH STEMS
    noteheads_with_stems = combine_noteheads_with_stems(
        symbols.noteheads, symbols.stems_rests
    )
    logger.info(f"Found {len(noteheads_with_stems)} noteheads")
    if len(noteheads_with_stems) == 0:
        raise Exception("No noteheads found")

    avg_notehead_height = float(
        np.median([n.notehead.size[1] for n in noteheads_with_stems])
    )
    logger.info(f"Average notehead height: {avg_notehead_height}")

    write_debug_image(image, "14_notes_with_stems.png", drawables=noteheads_with_stems)

    ## DETECTING BAR LINES
    all_noteheads = [n.notehead for n in noteheads_with_stems]
    all_stems = [n.stem for n in noteheads_with_stems if n.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]

    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, avg_notehead_height)
    logger.info(f"Found {len(bar_line_boxes)} bar lines")

    write_debug_image(image, "15_bar_line_boxes.png", drawables=bar_line_boxes)

    ## DETECTING STAFFS
    staff_anchors = find_staff_anchors(
        symbols.staff_fragments, symbols.clefs_keys, are_clefs=True
    )
    logger.info(f"Found {len(staff_anchors)} clefs")

    possible_other_clefs = predict_other_anchors_from_clefs(
        staff_anchors, predictions.staff
    )
    logger.info(f"Found {len(possible_other_clefs)} possible other clefs")

    staff_anchors.extend(
        find_staff_anchors(
            symbols.staff_fragments, possible_other_clefs, are_clefs=True
        )
    )
    staff_anchors.extend(
        find_staff_anchors(symbols.staff_fragments, bar_line_boxes, are_clefs=False)
    )

    staff_anchors = filter_unusual_anchors(staff_anchors)
    logger.info(f"Found {len(staff_anchors)} staff anchors")

    write_debug_image(image, "16_staff_anchors.png", drawables=staff_anchors)

    raw_staffs_with_possible_dupes = find_raw_staffs_by_connecting_fragments(
        staff_anchors, symbols.staff_fragments
    )
    logger.info(f"Found {len(raw_staffs_with_possible_dupes)} staffs")

    raw_staffs = remove_duplicate_staffs(raw_staffs_with_possible_dupes)
    if len(raw_staffs_with_possible_dupes) != len(raw_staffs):
        logger.info(
            f"Removed {len(raw_staffs_with_possible_dupes) - len(raw_staffs)} duplicate staffs"
        )

    write_debug_image(image, "17_raw_staffs.png", drawables=raw_staffs)

    staffs = resample_staffs(raw_staffs)
    staffs = filter_edge_of_vision(staffs, image.shape)
    staffs = sorted(staffs, key=lambda staff: staff.min_y)  # sort top to bottom
    if len(staffs) == 0:
        raise Exception("No staffs found")

    global_unit_size = np.mean([staff.average_unit_size for staff in staffs])

    write_debug_image(image, "18_staffs.png", drawables=staffs)

    ## ADDING BAR LINES TO STAFFS
    bar_lines_found = add_bar_lines_to_staffs(staffs, bar_line_boxes)
    logger.info(f"Found {len(bar_lines_found)} bar lines")

    write_debug_image(image, "19_bar_lines.png", drawables=bar_lines_found)

    ## ADDING RESTS TO STAFFS
    possible_rests = [
        rest
        for rest in bar_lines_or_rests
        if not rest.is_overlapping_with_any(bar_line_boxes)
    ]
    rests = add_rests_to_staffs(staffs, possible_rests)
    logger.info(f"Found {len(rests)} rests")

    ## PREPARING BRACES AND DOTS
    all_classified = (
        predictions.notehead + predictions.clefs_keys + predictions.stems_rests
    )
    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    brace_dot = create_rotated_bboxes(
        brace_dot_img, skip_merging=True, max_size=(100, -1)
    )

    write_debug_image(image, "20_brace_dot_img.png", binary_map=brace_dot_img)

    ## ADDING NOTES TO STAFFS
    notes = add_notes_to_staffs(staffs, noteheads_with_stems, predictions.notehead)

    write_debug_image(image, "21_notes_on_staffs.png", drawables=notes)

    ## ADDING ACCIDENTALS TO STAFFS
    accidentals = add_accidentals_to_staffs(staffs, symbols.accidentals)
    logger.info(f"Found {len(accidentals)} accidentals")

    write_debug_image(image, "22_accidentals.png", drawables=accidentals)

    ## FINDING BRACES/BRACKETS/GRAND STAFF LINES
    multi_staffs = find_braces_brackets_and_grand_staff_lines(staffs, brace_dot)
    logger.info(
        f"Found {len(multi_staffs)} connected staffs (after merging grand staffs, multiple voices): "
        + f"{[len(staff.staffs) for staff in multi_staffs]}"
    )

    write_debug_image(image, "23_multi_staffs.png", drawables=multi_staffs)

    # PARSE STAFFS (RUN TROMR)
    result_staffs = parse_staffs(multi_staffs, predictions.preprocessed)

    # MAINTAIN ACCIDENTALS
    result_staffs = maintain_accidentals(result_staffs)

    if not is_first_page:
        result_staffs[0].measures[0].is_new_page = True

    return {"page": Page(result_staffs)}
    # pages.append(Page(result_staffs))


def generate_musicxml(pages: list[Page]) -> bytes:
    # xml_name = "result.musicxml"

    # MERGE STAFFS ACROSS PAGES BY VOICE
    total_staffs = merge_staffs_across_pages(pages)

    # GENERATE MUSICXML
    logger.info("Writing XML")
    xml = generate_xml(total_staffs)

    xml_string = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    xml_string += xml.to_string()
    file_data = xml_string.encode("utf-8")

    # logger.info(f"Writing results to {xml_name}")
    return file_data
    # return {
    #     "filename": xml_name,
    #     "file_data": file_data,
    # }
