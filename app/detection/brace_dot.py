import logging
import cv2
import numpy as np

from app.utils.constants import NDArray
from app.utils.bounding_box import RotatedBoundingBox
from app.classes.models import Staff, MultiStaff


logger = logging.getLogger(__name__)


def prepare_brace_dot_image(symbols: NDArray, staff: NDArray) -> NDArray:
    brace_dot = cv2.subtract(symbols, staff)
    # remove horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    out = cv2.erode(brace_dot.astype(np.uint8), kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    return cv2.dilate(out, kernel)


def filter_for_tall_elements(
    brace_dot: list[RotatedBoundingBox], staffs: list[Staff]
) -> list[RotatedBoundingBox]:
    """
    We filter elements in two steps:
    1. Use a rough unit size estimate to reduce the data size
    2. Find the closest staff and take its unit size to take warping into account
    """
    rough_unit_size = staffs[0].average_unit_size
    large_symbols = [
        symbol
        for symbol in brace_dot
        if symbol.size[1] > 2 * rough_unit_size and symbol.size[0] < 3 * rough_unit_size
    ]

    result = []
    for symbol in large_symbols:
        closest_staff = min(
            staffs, key=lambda staff: staff.y_distance_to(symbol.center)
        )
        unit_size = closest_staff.average_unit_size
        if symbol.size[1] > 4 * unit_size:
            result.append(symbol)
    return result


def get_connections_between_staffs_at_bar_lines(
    staff1: Staff, staff2: Staff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    """
    Returns symbols (likely brackets or braces) that connect both staffs via bar line overlaps.
    """
    bar_lines1 = staff1.get_bar_lines()
    bar_lines2 = staff2.get_bar_lines()
    result: list[RotatedBoundingBox] = []

    for symbol in brace_dot:
        symbol_thick = symbol.make_box_thicker(30)

        overlaps_staff1 = any(
            symbol_thick.is_overlapping(line.box) for line in bar_lines1
        )
        overlaps_staff2 = any(
            symbol_thick.is_overlapping(line.box) for line in bar_lines2
        )

        if overlaps_staff1 and overlaps_staff2:
            result.append(symbol)

    return result


def get_connections_between_staffs_at_clefs(
    staff1: Staff, staff2: Staff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    """
    Returns symbols (likely brackets or braces) that connect both staffs via overlapping clefs.
    """
    clefs1 = staff1.get_clefs()
    clefs2 = staff2.get_clefs()
    result: list[RotatedBoundingBox] = []
    for symbol in brace_dot:
        overlaps_staff1 = any(symbol.is_overlapping(clef.box) for clef in clefs1)
        overlaps_staff2 = any(symbol.is_overlapping(clef.box) for clef in clefs2)

        if overlaps_staff1 and overlaps_staff2:
            result.append(symbol)

    return result


def get_connections_between_staffs_at_lines(
    staff1: Staff, staff2: Staff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    """
    Returns symbols (likely brackets or braces) that connect both staffs via overlapping staff lines.
    """
    result: list[RotatedBoundingBox] = []
    tolerance_for_touching_clefs = int(round(staff1.average_unit_size * 2))
    for symbol in brace_dot:
        symbol_thick = symbol.make_box_thicker(tolerance_for_touching_clefs)

        point1 = staff1.get_at(symbol.center[0])
        point2 = staff2.get_at(symbol.center[0])
        if point1 is None or point2 is None:
            continue

        if symbol_thick.is_overlapping(
            point1.to_bounding_box()
        ) and symbol_thick.is_overlapping(point2.to_bounding_box()):
            result.append(symbol)

    return result


def get_connections_between_staffs(
    staff1: Staff, staff2: Staff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    result = []
    result.extend(
        get_connections_between_staffs_at_bar_lines(staff1, staff2, brace_dot)
    )
    result.extend(get_connections_between_staffs_at_clefs(staff1, staff2, brace_dot))
    result.extend(get_connections_between_staffs_at_lines(staff1, staff2, brace_dot))
    return result


def merge_multi_staff_if_they_share_a_staff(
    staffs: list[MultiStaff],
) -> list[MultiStaff]:
    """
    If two MultiStaff objects share a staff, merge them into one MultiStaff object.
    """
    result: list[MultiStaff] = []
    for staff in staffs:
        any_merged = False
        for existing in result:
            if set(staff.staffs) & set(existing.staffs):
                result.remove(existing)
                result.append(existing.merge(staff))
                any_merged = True
                break
        if not any_merged:
            result.append(staff)
    return result


def find_braces_brackets_and_grand_staff_lines(
    staffs: list[Staff], brace_dot: list[RotatedBoundingBox]
) -> list[MultiStaff]:
    """
    Connect staffs from multiple voices or grand staffs by searching for brackets and grand staffs.
    """
    brace_dot = filter_for_tall_elements(brace_dot, staffs)
    result = []
    minimum_connections_to_form_combined_staff = 1

    for i, staff in enumerate(staffs):
        neighbors: list[Staff] = []
        if i > 0:
            neighbors.append(staffs[i - 1])
        if i < len(staffs) - 1:
            neighbors.append(staffs[i + 1])

        any_connected_neighbor = False
        for neighbor in neighbors:
            connections = get_connections_between_staffs(staff, neighbor, brace_dot)
            if len(connections) >= minimum_connections_to_form_combined_staff:
                result.append(MultiStaff([staff, neighbor], connections))
                any_connected_neighbor = True

        if not any_connected_neighbor:
            result.append(MultiStaff([staff], []))

    return merge_multi_staff_if_they_share_a_staff(result)
