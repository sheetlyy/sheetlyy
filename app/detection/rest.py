import logging
import numpy as np

from app.classes.models import Rest, Staff
from app.utils.bounding_box import RotatedBoundingBox


logger = logging.getLogger(__name__)


def add_rests_to_staffs(
    staffs: list[Staff], rests: list[RotatedBoundingBox]
) -> list[Rest]:
    result = []
    central_staff_line_idxs = [1, 2]
    for staff in staffs:
        for rest in rests:
            if not staff.is_on_staff_zone(rest):
                continue

            point = staff.get_at(rest.center[0])
            if point is None:
                continue

            center = rest.center
            idx_of_closest_y = np.argmin(np.abs([y - center[1] for y in point.y]))
            is_in_center = idx_of_closest_y in central_staff_line_idxs
            if not is_in_center:
                continue

            rest_w, rest_h = rest.size
            min_size = 0.7 * point.average_unit_size
            max_size = 3.5 * point.average_unit_size
            if not (min_size <= rest_w <= max_size and min_size <= rest_h <= max_size):
                continue

            bbox = rest.to_bounding_box()
            rest_symbol = Rest(bbox)
            staff.add_symbol(rest_symbol)
            result.append(rest_symbol)
    return result
