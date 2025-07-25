import logging
from worker.classes.models import BarLine, Staff
from worker.utils.bounding_box import RotatedBoundingBox


logger = logging.getLogger(__name__)


def detect_bar_lines(
    bar_lines: list[RotatedBoundingBox], unit_size: float
) -> list[RotatedBoundingBox]:
    """
    Detects bar lines by filtering candidates based on size.
    """
    min_height = 3 * unit_size
    max_width = 2 * unit_size

    return [
        bar_line
        for bar_line in bar_lines
        if bar_line.size[1] >= min_height and bar_line.size[0] <= max_width
    ]


def add_bar_lines_to_staffs(
    staffs: list[Staff], bar_lines: list[RotatedBoundingBox]
) -> list[BarLine]:
    result = []
    for staff in staffs:
        for bar_line in bar_lines:
            if not staff.is_on_staff_zone(bar_line):
                continue

            point = staff.get_at(bar_line.center[0])
            if point is None:
                continue

            bar_line_to_staff_tolerance = 4 * point.average_unit_size
            if abs(bar_line.top_left[1] - point.y[0]) > bar_line_to_staff_tolerance:
                continue
            if abs(bar_line.bottom_left[1] - point.y[-1]) > bar_line_to_staff_tolerance:
                continue

            bar_line_symbol = BarLine(bar_line)
            staff.add_symbol(bar_line_symbol)
            result.append(bar_line_symbol)
    return result
