import logging
from app.utils.bounding_box import RotatedBoundingBox
from app.classes.models import Staff, Accidental


logger = logging.getLogger(__name__)


def add_accidentals_to_staffs(
    staffs: list[Staff], accidentals: list[RotatedBoundingBox]
) -> list[Accidental]:
    result = []
    for staff in staffs:
        for accidental in accidentals:
            # validate position
            if not staff.is_on_staff_zone(accidental):
                continue

            point = staff.get_at(accidental.center[0])
            if point is None:
                continue

            # validate size
            w, h = accidental.size
            min_size = 0.5 * staff.average_unit_size
            max_size = 3 * staff.average_unit_size
            if any(dim < min_size or dim > max_size for dim in (w, h)):
                continue

            position = point.find_position_in_unit_sizes(accidental)
            accidental_bbox = accidental.to_bounding_box()
            symbol = Accidental(accidental_bbox, position)
            staff.add_symbol(symbol)
            result.append(symbol)
    return result
