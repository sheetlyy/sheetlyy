import logging
import cv2.typing as cvt
import numpy as np
from dataclasses import dataclass
from typing import Optional

from app.utils.bounding_box import (
    DebugDrawable,
    BoundingEllipse,
    RotatedBoundingBox,
)
from app.utils.constants import NDArray
from app.classes.models import StemDirection, Staff, Note, NoteGroup, SymbolOnStaff


logger = logging.getLogger(__name__)


@dataclass
class NoteheadWithStem(DebugDrawable):
    notehead: BoundingEllipse
    stem: Optional[RotatedBoundingBox]
    stem_direction: Optional[StemDirection] = None

    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)
    ) -> None:
        self.notehead.draw_onto_image(img, color)
        if self.stem is not None:
            self.stem.draw_onto_image(img, color)


def combine_noteheads_with_stems(
    noteheads: list[BoundingEllipse],
    stems: list[RotatedBoundingBox],
) -> list[NoteheadWithStem]:
    """
    Combines noteheads with their stems as this lets us differentiate between stems and bar lines.
    """
    result = []

    # sort from top to bottom
    noteheads = sorted(noteheads, key=lambda n: n.center[1])

    for notehead in noteheads:
        thickened_notehead = notehead.make_box_thicker(15)
        matched_stem = None

        for stem in stems:
            if stem.is_overlapping(thickened_notehead):
                matched_stem = stem
                break

        if matched_stem:
            is_stem_above = matched_stem.center[1] < notehead.center[1]
            direction = StemDirection.UP if is_stem_above else StemDirection.DOWN
            result.append(NoteheadWithStem(notehead, matched_stem, direction))
        else:
            result.append(NoteheadWithStem(notehead, None, None))

    return result


def get_center(bbox: cvt.Rect) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    cen_y = int(round((y1 + y2) / 2))
    cen_x = int(round((x1 + x2) / 2))
    return cen_x, cen_y


def adjust_bbox(bbox: cvt.Rect, noteheads: NDArray) -> cvt.Rect:
    x1, y1, x2, y2 = bbox
    region = noteheads[y1:y2, x1:x2]
    ys, _ = np.where(region > 0)
    if len(ys) == 0:
        # invalid note, will be eliminated with 0 height
        return bbox
    top = np.min(ys) + y1 - 1
    bottom = np.max(ys) + y1 + 1
    return (x1, int(top), x2, int(bottom))


def check_bbox_size(
    bbox: cvt.Rect, noteheads: NDArray, unit_size: float
) -> list[cvt.Rect]:
    x1, y1, x2, y2 = bbox

    # actual note size
    w = x2 - x1
    h = y2 - y1
    cen_x, _ = get_center(bbox)

    # expected note size
    notehead_size_ratio = 1.285714  # width/height
    expected_w = notehead_size_ratio * unit_size
    expected_h = unit_size

    result: list[cvt.Rect] = []
    if abs(w - expected_w) > abs(w - expected_w * 2):
        # contains at least 2 notes, 1 left and 1 right
        left_box: cvt.Rect = (x1, y1, cen_x, y2)
        right_box: cvt.Rect = (cen_x, y1, x2, y2)

        # upper and lower bounds could have changed
        left_box = adjust_bbox(left_box, noteheads)
        right_box = adjust_bbox(right_box, noteheads)

        # check recursively
        if left_box is not None:
            result.extend(check_bbox_size(left_box, noteheads, unit_size))
        if right_box is not None:
            result.extend(check_bbox_size(right_box, noteheads, unit_size))

    # check height
    if len(result) > 0:
        result = [
            b for box in result for b in check_bbox_size(box, noteheads, unit_size)
        ]
    else:
        num_notes = int(round(h / expected_h))
        if num_notes > 0:
            sub_h = h // num_notes
            for i in range(num_notes):
                sub_box = (
                    x1,
                    round(y1 + i * sub_h),
                    x2,
                    round(y1 + (i + 1) * sub_h),
                )
                result.append(sub_box)

    return result


def split_clumps_of_noteheads(
    notehead: NoteheadWithStem, noteheads: NDArray, staff: Staff
) -> list[NoteheadWithStem]:
    """
    Noteheads might be clumped together by the notehead detection algorithm.
    """
    bbox = [
        int(notehead.notehead.top_left[0]),
        int(notehead.notehead.top_left[1]),
        int(notehead.notehead.bottom_right[0]),
        int(notehead.notehead.bottom_right[1]),
    ]
    split_boxes = check_bbox_size(bbox, noteheads, staff.average_unit_size)  # type: ignore
    if len(split_boxes) <= 1:
        return [notehead]

    result = []
    for box in split_boxes:
        x1, y1, x2, y2 = box
        center = get_center(box)
        size = (x2 - x1, y2 - y1)
        new_note = NoteheadWithStem(
            BoundingEllipse((center, size, 0), notehead.notehead.contours),
            notehead.stem,
            notehead.stem_direction,
        )
        result.append(new_note)
    return result


def are_notes_likely_a_chord(note1: Note, note2: Note, tolerance: float) -> bool:
    if note1.stem is None or note2.stem is None:
        return abs(note1.center[0] - note2.center[0]) < tolerance
    return abs(note1.stem.center[0] - note2.stem.center[0]) < tolerance


def create_note_group(notes: list[Note]) -> Note | NoteGroup:
    if len(notes) == 1:
        return notes[0]
    return NoteGroup(notes)


def group_notes_on_staff(staff: Staff) -> None:
    notes = staff.get_notes()
    groups: list[list[Note]] = []

    for note in notes:
        group_found = False
        for group in groups:
            # check if this note belongs to any note in group
            if any(
                are_notes_likely_a_chord(
                    note, grouped_note, staff.average_unit_size  # type: ignore
                )
                for grouped_note in group
            ):
                group.append(note)
                group_found = True
                break
        if not group_found:
            groups.append([note])

    note_groups: list[SymbolOnStaff] = [create_note_group(group) for group in groups]
    note_groups.extend(staff.get_all_except_notes())
    # sort by x
    staff.symbols = sorted(note_groups, key=lambda group: group.center[0])


def add_notes_to_staffs(
    staffs: list[Staff], noteheads: list[NoteheadWithStem], notehead_pred: NDArray
) -> list[Note]:
    result = []
    for staff in staffs:
        for notehead_chunk in noteheads:
            # validate position
            if not staff.is_on_staff_zone(notehead_chunk.notehead):
                continue

            center = notehead_chunk.notehead.center
            point = staff.get_at(center[0])
            if point is None:
                continue

            # validate size
            notehead_chunk_w, notehead_chunk_h = notehead_chunk.notehead.size
            if (
                notehead_chunk_w < 0.5 * point.average_unit_size
                or notehead_chunk_h < 0.5 * point.average_unit_size
            ):
                continue

            # split and validate each notehead
            for note in split_clumps_of_noteheads(notehead_chunk, notehead_pred, staff):
                # validate split notehead position
                point = staff.get_at(note.notehead.center[0])
                if point is None:
                    continue

                # validate split notehead size
                note_w, note_h = note.notehead.size
                if (
                    note_w < 0.5 * point.average_unit_size
                    or note_w > 3 * point.average_unit_size
                    or note_h < 0.5 * point.average_unit_size
                    or note_h > 2 * point.average_unit_size
                ):
                    continue

                position = point.find_position_in_unit_sizes(note.notehead)
                new_note = Note(note.notehead, position, note.stem, note.stem_direction)
                result.append(new_note)
                staff.add_symbol(new_note)

    # group notes into chords and update staff symbols
    total_notes = 0
    total_groups = 0
    for staff in staffs:
        group_notes_on_staff(staff)
        total_notes += len(staff.get_notes())
        total_groups += len(staff.get_note_groups())

    logger.info(
        f"After grouping there are {total_notes} notes and {total_groups} note groups"
    )
    return result
