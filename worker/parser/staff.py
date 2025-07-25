import logging
import cv2
import numpy as np
from typing import Optional

from worker.transformer.configs import default_config
from worker.classes.models import MultiStaff, Staff, NoteGroup, Note
from worker.classes.results import (
    ResultStaff,
    ResultMeasure,
    ResultClef,
    ResultChord,
    move_pitch_to_clef,
    ResultTimeSignature,
    Page,
)
from worker.utils.constants import NDArray
from worker.detection.staff_dewarp import StaffDewarping, dewarp_staff_image
from worker.parser.staff_tromr import parse_staff_tromr


logger = logging.getLogger(__name__)


def have_all_the_same_number_of_staffs(staffs: list[MultiStaff]) -> bool:
    return all(len(staff.staffs) == len(staffs[0].staffs) for staff in staffs)


def is_close_to_image_top_or_bottom(staff: MultiStaff, image: NDArray) -> bool:
    tolerance = 50
    closest_distance_to_top_or_bottom = [
        min(s.min_x, image.shape[0] - s.max_x) for s in staff.staffs
    ]
    return min(closest_distance_to_top_or_bottom) < tolerance


def ensure_same_number_of_staffs(
    staffs: list[MultiStaff], image: NDArray
) -> list[MultiStaff]:
    # check if all have the same number of staffs
    if have_all_the_same_number_of_staffs(staffs):
        return staffs

    if len(staffs) > 2:
        if is_close_to_image_top_or_bottom(
            staffs[0], image
        ) and have_all_the_same_number_of_staffs(staffs[1:]):
            logger.info(
                "Removing first system from all voices, as it has a different number of staffs"
            )
            return staffs[1:]

        if is_close_to_image_top_or_bottom(
            staffs[-1], image
        ) and have_all_the_same_number_of_staffs(staffs[:-1]):
            logger.info(
                "Removing last system from all voices, as it has a different number of staffs"
            )
            return staffs[:-1]

    result: list[MultiStaff] = []
    for staff in staffs:
        result.extend(staff.break_apart())

    return sorted(result, key=lambda s: s.staffs[0].min_y)


def determine_ranges(staffs: list[MultiStaff]) -> list[float]:
    staff_centers = [
        (staff.max_y + staff.min_y) // 2 for voice in staffs for staff in voice.staffs
    ]
    return sorted(staff_centers)


def get_min_man_y_pos_of_notes(staff: Staff) -> tuple[float, float]:
    if len(staff.symbols) == 0:
        min_y = staff.min_y - 3.5 * staff.average_unit_size
        max_y = staff.max_y + 3.5 * staff.average_unit_size
        return min_y, max_y  # type: ignore

    min_y = staff.min_y - 2.5 * staff.average_unit_size
    max_y = staff.max_y + 2.5 * staff.average_unit_size
    for symbol in staff.symbols:
        if isinstance(symbol, NoteGroup):
            for note in symbol.notes:
                y = note.center[1]
                min_y = min(min_y, y - staff.average_unit_size)
                max_y = max(max_y, y + staff.average_unit_size)
        elif isinstance(symbol, Note):
            y = symbol.center[1]
            min_y = min(min_y, y - staff.average_unit_size)
            max_y = max(max_y, y + staff.average_unit_size)
    return min_y, max_y  # type: ignore


def calculate_region(staff: Staff, x_values: NDArray, y_values: NDArray) -> NDArray:
    if len(x_values) == 0 or len(y_values) == 0:
        x_min = staff.min_x - 2 * staff.average_unit_size
        x_max = staff.max_x + 2 * staff.average_unit_size
        y_min, y_max = get_min_man_y_pos_of_notes(staff)
    else:
        x_min = min(*x_values, staff.min_x) - 2 * staff.average_unit_size
        x_max = max(*x_values, staff.max_x) + 2 * staff.average_unit_size
        staff_min_y, staff_max_y = get_min_man_y_pos_of_notes(staff)
        y_min = min(*(y_values - 0.5 * staff.average_unit_size), staff_min_y)
        y_max = max(*(y_values + 0.5 * staff.average_unit_size), staff_max_y)
    return np.array([int(x_min), int(y_min), int(x_max), int(y_max)])


def calculate_offsets(staff: Staff, ranges: list[float]) -> list[float]:
    staff_center = (staff.max_y + staff.min_y) // 2
    y_offsets = []

    staff_above = max([r for r in ranges if r < staff_center], default=-1)
    if staff_above >= 0:
        y_offsets.append(staff.max_y - staff_above)

    staff_below = min([r for r in ranges if r > staff_center], default=-1)
    if staff_below >= 0:
        y_offsets.append(staff_below - staff.min_y)

    return y_offsets


def adjust_region(region: NDArray, y_offsets: list[float], staff: Staff) -> NDArray:
    if len(y_offsets) > 0:
        min_y_offset = min(y_offsets)
        if 3 * staff.average_unit_size < min_y_offset < 8 * staff.average_unit_size:
            region[1] = int(staff.min_y - min_y_offset)
            region[3] = int(staff.max_y + min_y_offset)
    return region


def get_tromr_canvas_size(
    image_shape: tuple[int, ...], margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    tromr_max_h_with_margin = default_config.max_height - margin_top - margin_bottom
    tromr_ratio = float(tromr_max_h_with_margin) / default_config.max_width
    h, w = image_shape[:2]

    # Calculate the new size such that it fits exactly into
    # default_config.max_height and default_config.max_width
    # while maintaining the aspect ratio of height and width.
    if h / w > tromr_ratio:
        # height is the limiting factor
        new_shape = [
            int(w / h * tromr_max_h_with_margin),
            tromr_max_h_with_margin,
        ]
    else:
        # width is the limiting factor
        new_shape = [default_config.max_width, int(h / w * default_config.max_width)]
    return np.array(new_shape)


def crop_image_and_return_new_top(
    image: NDArray, x1: float, y1: float, x2: float, y2: float
) -> tuple[NDArray, NDArray]:
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    x1_limited = max(0, min(image.shape[1] - 1, int(round(x_min))))
    y1_limited = max(0, min(image.shape[0] - 1, int(round(y_min))))
    x2_limited = max(0, min(image.shape[1] - 1, int(round(x_max))))
    y2_limited = max(0, min(image.shape[0] - 1, int(round(y_max))))
    new_top_x = np.array([x1_limited, y1_limited])
    return image[y1_limited:y2_limited, x1_limited:x2_limited], new_top_x


def dewarp_staff(
    staff: Staff, dewarp: Optional[StaffDewarping], region: NDArray, scaling: float
) -> Staff:
    """
    Applies the same transformation on the staff coordinates as we did on the image.
    """

    def transform_coordinates(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point[0] - region[0], point[1] - region[1]
        if dewarp is not None:
            x, y = dewarp.dewarp_point((x, y))
        return x * scaling, y * scaling

    return staff.transform_coordinates(transform_coordinates)


def remove_black_contours_at_edges_of_image(bgr: NDArray, unit_size: float) -> NDArray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 97, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    threshold = 2 * unit_size
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < threshold or h < threshold:
            continue

        is_at_edge_of_image = (
            x == 0 or y == 0 or x + w == bgr.shape[1] or y + h == bgr.shape[0]
        )
        if not is_at_edge_of_image:
            continue

        avg_gray_intensity = 127
        is_mostly_dark = np.mean(thresh[y : y + h, x : x + w]) < avg_gray_intensity
        if is_mostly_dark:
            continue

        bgr[y : y + h, x : x + w] = (255, 255, 255)
    return bgr


def center_image_on_canvas(
    image: NDArray, canvas_size: NDArray, margin_top: int = 0, margin_bottom: int = 0
) -> NDArray:
    resized = cv2.resize(image, canvas_size)  # type: ignore

    new_image = np.zeros(
        (default_config.max_height, default_config.max_width, 3), np.uint8
    )
    new_image[:, :] = (255, 255, 255)

    # copy resized image onto center of new image
    x_offset = 0
    tromr_max_h_with_margin = default_config.max_height - margin_top - margin_bottom
    y_offset = (tromr_max_h_with_margin - resized.shape[0]) // 2 + margin_top
    new_image[
        y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]
    ] = resized

    return new_image


def prepare_staff_image(
    index: int,
    ranges: list[float],
    staff: Staff,
    staff_image: NDArray,
    perform_dewarp: bool = True,
) -> tuple[NDArray, Staff]:
    centers = [s.center for s in staff.symbols]
    x_values = np.array([c[0] for c in centers])
    y_values = np.array([c[1] for c in centers])

    region = calculate_region(staff, x_values, y_values)
    y_offsets = calculate_offsets(staff, ranges)

    region = adjust_region(region, y_offsets, staff)
    image_dims = get_tromr_canvas_size(
        (int(region[3] - region[1]), int(region[2] - region[0]))
    )
    scaling_factor = image_dims[1] / (region[3] - region[1])
    staff_image = cv2.resize(
        staff_image,
        (
            int(staff_image.shape[1] * scaling_factor),
            int(staff_image.shape[0] * scaling_factor),
        ),
    )
    region = np.round(region * scaling_factor)

    if perform_dewarp:
        logger.info(f"Dewarping staff {index}")

        region_step1 = np.array(region) + np.array([-10, -50, 10, 50])
        staff_image, top_left = crop_image_and_return_new_top(
            staff_image, *region_step1
        )

        region_step2 = np.array(region) - np.array([*top_left, *top_left])
        top_left = top_left / scaling_factor
        staff = dewarp_staff(staff, None, top_left, scaling_factor)
        dewarp = dewarp_staff_image(staff_image, staff, index)

        staff_image = (255 * dewarp.dewarp(staff_image)).astype(np.uint8)
        staff_image, top_left = crop_image_and_return_new_top(
            staff_image, *region_step2
        )
        scaling_factor = 1

        logger.info(f"Dewarping staff {index} done")
    else:
        staff_image, top_left = crop_image_and_return_new_top(staff_image, *region)

    staff_image = remove_black_contours_at_edges_of_image(staff_image, staff.average_unit_size)  # type: ignore
    staff_image = center_image_on_canvas(staff_image, image_dims)

    return staff_image, staff


def parse_staff_image(
    ranges: list[float], index: int, staff: Staff, image: NDArray
) -> Optional[ResultStaff]:
    staff_image, transformed_staff = prepare_staff_image(index, ranges, staff, image)
    logger.info(f"Running TrOmr inference on staff {index}")
    result = parse_staff_tromr(staff_image=staff_image, staff=transformed_staff)
    return result


def remember_new_line(measures: list[ResultMeasure]) -> None:
    if len(measures) > 0:
        measures[0].is_new_line = True


def pick_dominant_clef(staff: ResultStaff) -> ResultStaff:
    clefs = [clef for clef in staff.get_symbols() if isinstance(clef, ResultClef)]
    clef_types = [clef.clef_type for clef in clefs]
    if len(clef_types) == 0:
        return staff

    most_frequent_clef_type = max(set(clef_types), key=clef_types.count)
    if most_frequent_clef_type is None:
        return staff

    if clef_types.count(most_frequent_clef_type) == 1:
        return staff

    circle_of_fifth = 0  # doesn't matter if we only look at the clef type
    most_frequent_clef = ResultClef(most_frequent_clef_type, circle_of_fifth)
    last_clef_was_originally = None

    for symbol in staff.get_symbols():
        if isinstance(symbol, ResultClef):
            last_clef_was_originally = ResultClef(symbol.clef_type, 0)
            symbol.clef_type = most_frequent_clef_type

        elif isinstance(symbol, ResultChord):
            for note in symbol.notes:
                note.pitch = move_pitch_to_clef(
                    note.pitch, last_clef_was_originally, most_frequent_clef
                )

        elif isinstance(symbol, ResultMeasure):
            for measure_symbol in symbol.symbols:
                if isinstance(measure_symbol, ResultClef):
                    last_clef_was_originally = ResultClef(measure_symbol.clef_type, 0)
                    measure_symbol.clef_type = most_frequent_clef_type

                elif isinstance(measure_symbol, ResultChord):
                    for note in measure_symbol.notes:
                        note.pitch = move_pitch_to_clef(
                            note.pitch, last_clef_was_originally, most_frequent_clef
                        )

    return staff


def pick_dominant_key_signature(staff: ResultStaff) -> ResultStaff:
    clefs = [clef for clef in staff.get_symbols() if isinstance(clef, ResultClef)]
    key_signatures = [clef.circle_of_fifth for clef in clefs]
    if len(key_signatures) == 0:
        return staff

    most_frequent_key = max(set(key_signatures), key=key_signatures.count)
    if most_frequent_key is None:
        return staff

    if key_signatures.count(most_frequent_key) == 1:
        return staff

    for clef in clefs:
        clef.circle_of_fifth = most_frequent_key

    return staff


def remove_redundant_clefs(measures: list[ResultMeasure]) -> None:
    last_clef = None
    for measure in measures:
        for symbol in measure.symbols:
            if isinstance(symbol, ResultClef):
                if last_clef is not None and last_clef == symbol:
                    measure.remove_symbol(symbol)
                else:
                    last_clef = symbol


def remove_all_but_first_time_signature(measures: list[ResultMeasure]) -> None:
    """
    The transformer tends to hallucinate time signatures. In most cases there is only one
    time signature at the beginning, so we remove all others.
    """
    last_sig = None
    for measure in measures:
        for symbol in measure.symbols:
            if isinstance(symbol, ResultTimeSignature):
                if last_sig is not None:
                    measure.remove_symbol(symbol)
                else:
                    last_sig = symbol


def merge_and_clean(
    staffs: list[ResultStaff], force_single_clef_type: bool
) -> ResultStaff:
    """
    Merge all staffs of a voice into a single staff.
    """
    result = ResultStaff([])
    for staff in staffs:
        result = result.merge(staff)
    if force_single_clef_type:
        pick_dominant_clef(result)

    pick_dominant_key_signature(result)
    remove_redundant_clefs(result.measures)
    remove_all_but_first_time_signature(result.measures)

    result.measures = [measure for measure in result.measures if not measure.is_empty()]
    return result


def parse_staffs(staffs: list[MultiStaff], image: NDArray) -> list[ResultStaff]:
    """
    Dewarps each staff and then runs it through an algorithm which extracts
    the rhythm and pitch information.
    """
    staffs = ensure_same_number_of_staffs(staffs, image)

    # for simplicity we call every staff in a MultiStaff a voice, even in a grand staff
    total_voices = len(staffs[0].staffs)
    i = 0
    ranges = determine_ranges(staffs)
    voices = []

    for voice in range(total_voices):
        staffs_for_voice = [staff.staffs[voice] for staff in staffs]
        result_for_voice: list[ResultStaff] = []
        for staff_idx, staff in enumerate(staffs_for_voice):
            result_staff = parse_staff_image(ranges, i, staff, image)

            if result_staff is None:
                logger.info(f"Staff was filtered out {i}")
                i += 1
                continue

            if result_staff.is_empty():
                logger.info(f"Skipping empty staff {i}")
                i += 1
                continue

            remember_new_line(result_staff.measures)
            result_for_voice.append(result_staff)
            i += 1

        # Piano music can have a change of clef, while for other instruments
        # we assume that the clef is the same for all staffs.
        # The number of voices is the only way we can distinguish between the two.
        force_single_clef_type = total_voices == 1
        voices.append(merge_and_clean(result_for_voice, force_single_clef_type))

    return voices


def merge_staffs_across_pages(pages: list[Page]) -> list[ResultStaff]:
    """
    Merges staffs by voice across pages.
    """
    voices_by_index = []
    num_voices = len(pages[0].staffs)  # should be 2 for piano

    for voice_idx in range(num_voices):
        staffs_for_voice = [page.staffs[voice_idx] for page in pages]
        merged_staff = merge_and_clean(staffs_for_voice, num_voices == 1)
        voices_by_index.append(merged_staff)

    return voices_by_index
