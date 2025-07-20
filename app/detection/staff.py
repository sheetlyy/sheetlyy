import logging
import cv2
import cv2.typing as cvt
import numpy as np
from typing import Optional, Iterable, Generator
from scipy import signal

from app.classes.models import StaffPoint, Staff
from app.utils.bounding_box import RotatedBoundingBox, DebugDrawable
from app.utils.constants import NDArray, max_line_gap_size, NUMBER_OF_LINES_ON_A_STAFF


logger = logging.getLogger(__name__)


class StaffLine(DebugDrawable):
    """
    Represents one staff line. Made up of multiple fragments.
    """

    def __init__(self, fragments: list[RotatedBoundingBox]):
        self.fragments = sorted(fragments, key=lambda f: f.center[0])

        self.min_x = min([frag.center[0] - frag.size[0] / 2 for frag in fragments])
        self.max_x = max([frag.center[0] + frag.size[0] / 2 for frag in fragments])
        self.min_y = min([frag.center[1] - frag.size[1] / 2 for frag in fragments])
        self.max_y = max([frag.center[1] + frag.size[1] / 2 for frag in fragments])

    def merge(self, other: "StaffLine") -> "StaffLine":
        fragments = self.fragments.copy()
        for other_fragment in other.fragments:
            if other_fragment not in fragments:
                fragments.append(other_fragment)
        return StaffLine(fragments)

    def get_at(self, x: float) -> Optional[RotatedBoundingBox]:
        """
        Gets the line fragment that is closest to the given x-value.
        """
        tolerance = 10
        for fragment in self.fragments:
            if (
                x >= fragment.center[0] - fragment.size[0] / 2 - tolerance
                and x <= fragment.center[0] + fragment.size[0] / 2 + tolerance
            ):
                return fragment
        return None

    def is_overlapping(self, other: "StaffLine") -> bool:
        for frag in self.fragments:
            for other_frag in other.fragments:
                if frag.is_overlapping(other_frag):
                    return True
        return False

    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)
    ) -> None:
        for fragment in self.fragments:
            fragment.draw_onto_image(img, color)


class StaffAnchor(DebugDrawable):
    """
    An anchor is what we call a reliable staff line. That is five parallel bar lines
    which by their relation to other symbols make it likely that they belong to a staff.
    This is a crucial step as it allows us to then build the complete staff.
    """

    def __init__(self, staff_lines: list[StaffLine], symbol: RotatedBoundingBox):
        self.staff_lines = staff_lines
        self.symbol = symbol

        # finds y-coords of the 5 lines at the x-pos of the symbol
        y_positions = sorted(
            [
                line.fragments[0].get_center_extrapolated(symbol.center[0])
                for line in staff_lines
            ]
        )
        # spacings between staff lines
        y_deltas = [
            abs(y_positions[i] - y_positions[i - 1]) for i in range(1, len(y_positions))
        ]
        self.unit_sizes = y_deltas
        self.average_unit_size = 0.0 if len(y_deltas) == 0 else float(np.mean(y_deltas))

        self.max_y = max([line.max_y for line in staff_lines])
        self.min_y = min([line.min_y for line in staff_lines])

        max_ledger_lines = 5
        # range for staff lines
        self.y_range = range(int(min(y_positions)), int(max(y_positions)))
        # range for staff lines + ledger lines
        self.zone = range(
            int(self.min_y - max_ledger_lines * self.average_unit_size),
            int(self.max_y + max_ledger_lines * self.average_unit_size),
        )

    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (0, 255, 0)
    ) -> None:
        for lines in self.staff_lines:
            lines.draw_onto_image(img, color)
        self.symbol.draw_onto_image(img, color)
        x = int(self.symbol.center[0])
        cv2.line(img, [x - 50, self.zone.start], [x + 50, self.zone.start], color, 2)
        cv2.line(img, [x - 50, self.zone.stop], [x + 50, self.zone.stop], color, 2)


def break_wide_fragments(
    fragments: list[RotatedBoundingBox],
    limit: int = 100,
) -> list[RotatedBoundingBox]:
    """
    Wide fragments (large x dimension) which are curved tend to be filtered by later steps.
    We instead split them into smaller parts, so that the parts better approximate the different
    angles of the curve.
    """
    result = []
    for fragment in fragments:
        remaining_fragment = fragment
        while remaining_fragment.size[0] > limit:  # size[0] = width
            min_x = min(c[0][0] for c in remaining_fragment.contours)
            contours_left = [
                c for c in remaining_fragment.contours if c[0][0] < min_x + limit
            ]
            contours_right = [
                c for c in remaining_fragment.contours if c[0][0] >= min_x + limit
            ]

            # sort by x
            contours_left = sorted(contours_left, key=lambda c: c[0][0])
            contours_right = sorted(contours_right, key=lambda c: c[0][0])
            if len(contours_left) == 0 or len(contours_right) == 0:
                break

            # make sure contours remain connected by adding
            # first point of right side to left side and vice versa
            contours_left.append(contours_right[0])
            contours_right.append(contours_left[-1])

            left_box = cv2.minAreaRect(np.array(contours_left))
            right_box = cv2.minAreaRect(np.array(contours_right))

            result.append(RotatedBoundingBox(left_box, np.array(contours_left)))
            remaining_fragment = RotatedBoundingBox(right_box, np.array(contours_right))

        result.append(remaining_fragment)

    return result


def connect_staff_lines(
    staff_fragments: list[RotatedBoundingBox], unit_size: float
) -> list[StaffLine]:
    """
    Checks which fragments connect to each other (extrapolation is used to fill gaps)
    and builds a list of StaffLines.
    """
    # we sort right to left so that pop() retrieves items from left to right
    fragments_right_to_left = sorted(
        staff_fragments, key=lambda f: f.bottom_left[0], reverse=True
    )
    result: list[list[RotatedBoundingBox]] = []
    active_lines: list[list[RotatedBoundingBox]] = []
    last_cleanup_x: float = 0

    while len(fragments_right_to_left) > 0:
        current_fragment: RotatedBoundingBox = fragments_right_to_left.pop()
        x = current_fragment.bottom_left[0]

        # removes lines that are too far behind to possibly connect with the current one
        if x - last_cleanup_x > max_line_gap_size(unit_size):
            active_lines = [
                line
                for line in active_lines
                if x - line[-1].bottom_right[0] < max_line_gap_size(unit_size)
            ]
            last_cleanup_x = x

        # skip very short fragments
        if current_fragment.size[0] < unit_size / 5:
            continue

        # try to connect with active lines
        connected = False
        for line in active_lines:
            if line[-1].is_overlapping_extrapolated(current_fragment, unit_size):
                line.append(current_fragment)
                connected = True
                # break

        # if not connected, start new group
        if not connected:
            new_line = [current_fragment]
            result.append(new_line)
            active_lines.append(new_line)

    result_top_to_bottom = sorted(result, key=lambda frags: frags[0].center[1])
    return [StaffLine(fragments) for fragments in result_top_to_bottom]


def find_staff_anchors(
    staff_fragments: list[RotatedBoundingBox],
    anchor_symbols: list[RotatedBoundingBox],
    are_clefs: bool = False,
) -> list[StaffAnchor]:
    """
    Finds staff anchors by looking for five parallel lines which go
    over or interrupt symbols which are always on staffs
    (and never above or beyond them like notes can be).
    """
    result: list[StaffAnchor] = []

    for center_symbol in anchor_symbols:
        # As the symbol disconnects the staff lines it's the hardest to detect them at the center.
        # Therefore we try to detect them at the left and right side of the symbol as well.
        if are_clefs:
            adjacent = [
                center_symbol,
                center_symbol.move_x_horizontal_by(50),
                center_symbol,
                center_symbol.move_x_horizontal_by(100),
                center_symbol,
                center_symbol.move_x_horizontal_by(150),
            ]
        else:
            adjacent = [
                center_symbol.move_x_horizontal_by(-10),
                center_symbol.move_x_horizontal_by(-5),
                center_symbol,
                center_symbol.move_x_horizontal_by(5),
                center_symbol.move_x_horizontal_by(10),
            ]

        for symbol in adjacent:
            estimated_unit_size = round(symbol.size[1] / NUMBER_OF_LINES_ON_A_STAFF - 1)

            # find fragments that overlap anchor symbol
            thickened_symbol = symbol.make_taller_by(estimated_unit_size)
            overlapping_fragments = [
                f for f in staff_fragments if f.is_intersecting(thickened_symbol)
            ]

            # connect fragments into staff lines
            connected_lines = connect_staff_lines(
                overlapping_fragments, estimated_unit_size
            )
            is_short_connected_line = 2 * estimated_unit_size
            if len(connected_lines) > NUMBER_OF_LINES_ON_A_STAFF:
                # filter out short staff line segments
                connected_lines = [
                    line
                    for line in connected_lines
                    if (line.max_x - line.min_x) > is_short_connected_line
                ]
            if not len(connected_lines) == NUMBER_OF_LINES_ON_A_STAFF:
                continue

            # check if staff lines are parallel
            are_lines_parallel = True
            all_angles = []
            all_fragments: list[RotatedBoundingBox] = []
            for line in connected_lines:
                for fragment in line.fragments:
                    all_angles.append(fragment.angle)
                    all_fragments.append(fragment)
            if len(all_angles) == 0:
                continue

            average_angle = np.mean(all_angles)
            max_angle_for_lines_to_be_parallel = 5
            for fragment in all_fragments:
                if (
                    abs(fragment.angle - average_angle)
                    > max_angle_for_lines_to_be_parallel
                    and fragment.size[0] > is_short_connected_line
                ):
                    are_lines_parallel = False
                    break
            if not are_lines_parallel:
                continue

            # check if lines are crossing
            are_lines_crossing = False
            for i in range(len(connected_lines)):
                for j in range(i + 1, len(connected_lines)):
                    if connected_lines[i].is_overlapping(connected_lines[j]):
                        are_lines_crossing = True
                        break
            if are_lines_crossing:
                continue

            # check if begins or ends on one staff line
            if not are_clefs:
                begins_or_ends_on_one_staff_line = False
                for staff_line in connected_lines:
                    fragment = staff_line.get_at(symbol.center[0])
                    if fragment is None:
                        continue
                    staff_y = fragment.get_center_extrapolated(symbol.center[0])
                    if abs(staff_y - symbol.center[1]) < estimated_unit_size:
                        begins_or_ends_on_one_staff_line = True
                        break
                if not begins_or_ends_on_one_staff_line:
                    continue

            result.append(StaffAnchor(connected_lines, symbol))
    return result


def filter_line_peaks(
    peaks: NDArray,
    max_gap_ratio: float = 1.5,
) -> list[int]:
    """
    Assigns group IDs to peaks. Returns a list of integers assigning a group to each peak
    (same length as `peaks`).
    """
    if len(peaks) == 0:
        return []

    # filter by x-axis
    gaps = peaks[1:] - peaks[:-1]
    count = max(5, round(len(peaks) * 0.2))
    approx_unit = np.mean(np.sort(gaps)[:count])
    max_gap = approx_unit * max_gap_ratio

    # prepend an invalid peak for better handling edge case
    ext_peaks = [peaks[0] - max_gap - 1] + list(peaks)
    group_ids = []
    curr_group_id = -1

    for i in range(1, len(ext_peaks)):
        if ext_peaks[i] - ext_peaks[i - 1] > max_gap:
            curr_group_id += 1
        group_ids.append(curr_group_id)

    return group_ids


def find_horizontal_lines(
    vertical_slice: NDArray,
    unit_size: float,
    line_threshold: float = 0.0,
) -> list[list[int]]:
    """
    Detects horizontal staff lines in a vertical image slice.
    Returns a list of staffs as line groups, each represented by a list of 5 y-coordinates
    of the staff lines in the group.
    """
    # count intensity per row (y)
    row_intensity = np.zeros(len(vertical_slice), dtype=np.uint16)
    sub_ys, _ = np.where(vertical_slice > 0)
    for y in sub_ys:
        row_intensity[y] += 1

    # normalize and find peaks (potential staff lines)
    row_intensity = np.insert(
        row_intensity, [0, len(row_intensity)], [0, 0]  # prepend/append 0s
    )
    norm = (row_intensity - np.mean(row_intensity)) / np.std(row_intensity)
    line_peaks, _ = signal.find_peaks(
        norm, height=line_threshold, distance=unit_size, prominence=1
    )
    line_peaks -= 1
    norm = norm[1:-1]  # remove prepended/appended 0s

    # group peaks into potential staffs
    group_ids = filter_line_peaks(line_peaks)
    staff_line_groups: dict[int, list[int]] = {}
    for i, peak in enumerate(line_peaks):
        gid = group_ids[i]
        if gid not in staff_line_groups:
            staff_line_groups[gid] = []
        staff_line_groups[gid].append(peak)

    # filter to only complete staffs (5 lines)
    line_groups = [
        sorted(lines)
        for lines in staff_line_groups.values()
        if len(lines) == NUMBER_OF_LINES_ON_A_STAFF
    ]
    return line_groups


def predict_other_anchors_from_clefs(
    clef_anchors: list[StaffAnchor],
    image: NDArray,
) -> list[RotatedBoundingBox]:
    if len(clef_anchors) == 0:
        return []

    average_unit_size = float(np.mean([a.average_unit_size for a in clef_anchors]))
    clefs = [a.symbol for a in clef_anchors]

    # create horizontal zones around clefs (increase range right of clef to find staff lines)
    margin_right = 10
    ranges = [
        range(
            max(int(c.symbol.bottom_left[0]), 0),
            min(int(c.symbol.top_right[0] + margin_right), image.shape[1]),
        )
        for c in clef_anchors
    ]
    ranges = sorted(ranges, key=lambda r: r.start)

    # merge overlapping zones
    clef_zones: list[range] = []
    for i, r in enumerate(ranges):
        if i == 0:
            clef_zones.append(r)
        else:
            overlaps_with_the_last = r.start < clef_zones[-1].stop
            if overlaps_with_the_last:
                clef_zones[-1] = range(clef_zones[-1].start, r.stop)
            else:
                clef_zones.append(r)

    result: list[RotatedBoundingBox] = []
    for zone in clef_zones:
        vertical_slice = image[:, zone]
        line_groups = find_horizontal_lines(vertical_slice, average_unit_size)

        for group in line_groups:
            min_y, max_y = min(group), max(group)
            center_x = zone.start + (zone.stop - zone.start) / 2
            center_y = (min_y + max_y) / 2
            box = (
                (int(center_x), int(center_y)),
                (zone.stop - zone.start, int(max_y - min_y)),
                0,
            )
            result.append(RotatedBoundingBox(box, np.array([])))
    return [b for b in result if not b.is_overlapping_with_any(clefs)]


def filter_unusual_anchors(anchors: list[StaffAnchor]) -> list[StaffAnchor]:
    """
    Filters anchors by unit size.
    """
    if len(anchors) == 0:
        return anchors

    unit_sizes = [a.average_unit_size for a in anchors]
    average = np.mean(unit_sizes)
    std_dev = np.std(unit_sizes)

    return [
        anchor
        for anchor in anchors
        if abs(anchor.average_unit_size - average) <= 2 * std_dev
    ]


class RawStaff(RotatedBoundingBox):
    """
    A raw staff is made of parts which we found in the image. It has gaps, and segments start and
    end differently on every staff line.
    """

    def __init__(
        self, staff_id: int, lines: list[StaffLine], anchors: list[StaffAnchor]
    ):
        contours = self._get_all_contours(lines)
        box = cv2.minAreaRect(np.array(contours))
        super().__init__(box, np.concatenate(contours))

        self.staff_id = staff_id
        self.lines = lines
        self.anchors = anchors

        self.min_x = self.center[0] - self.size[0] / 2
        self.max_x = self.center[0] + self.size[0] / 2
        self.min_y = self.center[1] - self.size[1] / 2
        self.max_y = self.center[1] + self.size[1] / 2

    def merge(self, other: "RawStaff") -> "RawStaff":
        lines: list[StaffLine] = []
        for i, line in enumerate(self.lines):
            lines.append(other.lines[i].merge(line))
        return RawStaff(self.staff_id, lines, self.anchors + other.anchors)

    def _get_all_contours(self, lines: list[StaffLine]) -> list[cvt.MatLike]:
        contours: list[cvt.MatLike] = []
        for line in lines:
            for fragment in line.fragments:
                contours.extend(fragment.contours)
        return contours

    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)
    ) -> None:
        for line in self.lines:
            line.draw_onto_image(img, color)


def get_staff_for_anchor(
    anchor: StaffAnchor, staffs: list[RawStaff]
) -> Optional[RawStaff]:
    for staff in staffs:
        for i, anchor_line in enumerate(anchor.staff_lines):
            anchor_fragments = set(anchor_line.fragments)
            staff_fragments = set(staff.lines[i].fragments)
            if anchor_fragments.issubset(staff_fragments):
                return staff
    return None


def find_raw_staffs_by_connecting_fragments(
    anchors: list[StaffAnchor],
    staff_fragments: list[RotatedBoundingBox],
) -> list[RawStaff]:
    """
    First we build a list of all lines by combining fragments. Then we identify the lines
    which go through the anchors and build staffs from them.
    """
    staffs: list[RawStaff] = []
    staff_id = 0
    for anchor in anchors:
        existing_staff = get_staff_for_anchor(anchor, staffs)

        zone_fragments = [
            fragment
            for fragment in staff_fragments
            if fragment.center[1] >= anchor.zone.start
            and fragment.center[1] <= anchor.zone.stop
        ]
        connected_lines = connect_staff_lines(zone_fragments, anchor.average_unit_size)

        staff_lines: list[StaffLine] = []
        for anchor_line in anchor.staff_lines:
            anchor_fragments = set(anchor_line.fragments)
            lines_that_fit_anchor_line = [
                line
                for line in connected_lines
                if anchor_fragments.issubset(set(line.fragments))
            ]
            if len(lines_that_fit_anchor_line) == 1:
                staff_lines.extend(lines_that_fit_anchor_line)
            else:
                staff_lines.append(anchor_line)

        if existing_staff:
            staffs.remove(existing_staff)
            staffs.append(
                existing_staff.merge(RawStaff(staff_id, staff_lines, [anchor]))
            )
        else:
            staffs.append(RawStaff(staff_id, staff_lines, [anchor]))

        staff_id += 1

    return staffs


def remove_duplicate_staffs(staffs: list[RawStaff]) -> list[RawStaff]:
    """
    Sometimes we find the same staff twice, but fail to connect them.
    This function removes the duplicates.
    """
    result: list[RawStaff] = []
    for staff in staffs:
        overlapping = [other for other in result if staff.is_overlapping(other)]
        if len(overlapping) == 0:
            # if no overlap with existing staffs, add new staff
            result.append(staff)
            continue

        if len(overlapping) >= 2:
            # if >= 2 overlaps, just skip this staff and take existing staffs for now
            continue

        # if 1 overlap, take the one with more anchors
        if len(overlapping[0].anchors) < len(staff.anchors):
            # staff with the most anchors is the most reliable one
            result = [s for s in result if s != overlapping[0]]
            result.append(staff)

    return result


def resample_staff_segment(
    anchor: StaffAnchor, staff: RawStaff, axis_range: Iterable[int]
) -> Generator[StaffPoint, None, None]:
    """
    Given an anchor, staff, and a range of x-coordinates, tracks and corrects the y-positions
    of the 5 staff lines over that range. Returns a cleaned sequence of StaffPoints, one for
    each valid x in the range.
    """
    anchor_x = anchor.symbol.center[0]
    line_fragments = [line.fragments[0] for line in anchor.staff_lines]
    y_centers: list[float] = [
        f.get_center_extrapolated(anchor_x) for f in line_fragments
    ]

    # dummy point at anchor points
    previous_point = StaffPoint(
        anchor_x, y_centers, float(np.mean([f.angle for f in line_fragments]))
    )

    for x in axis_range:
        lines = [line.get_at(x) for line in staff.lines]
        line_ys = [
            line.get_center_extrapolated(x) if line is not None else None
            for line in lines
        ]

        incomplete = all(y is None for y in line_ys)
        if incomplete:
            continue

        # invalidate lines that are not parallel
        valid_ys = [y for y in line_ys if y is not None]
        deltas = np.diff(valid_ys)
        for i, delta in enumerate(deltas):
            if delta < 0.5 * anchor.average_unit_size:
                line_ys[i] = None
                line_ys[i + 1] = None

        # invalidate lines that shift too far from prev position
        for i, prev_y in enumerate(previous_point.y):
            curr_y = line_ys[i]
            if (
                curr_y is not None
                and abs(curr_y - prev_y) > 0.5 * anchor.average_unit_size
            ):
                line_ys[i] = None

        # use extrapolation based on nearby valid lines to fill in missing values.
        # does a forward pass and a backward pass
        last_known_idx = -1
        for i in list(range(len(line_ys))) + list(reversed(list(range(len(line_ys))))):
            # track last known value
            if line_ys[i] is not None:
                last_known_idx = i
            # if value is missing, extrapolate
            elif last_known_idx >= 0:
                last_known_y = line_ys[last_known_idx]
                if last_known_y is not None:
                    line_ys[i] = last_known_y + anchor.average_unit_size * (
                        i - last_known_idx
                    )

        incomplete = any(y is None for y in line_ys)
        if incomplete:
            continue

        angle = float(np.mean([line.angle for line in lines if line is not None]))
        previous_point = StaffPoint(x, [y for y in line_ys if y is not None], angle)
        yield previous_point


def resample_staffs(staffs: list[RawStaff]) -> list[Staff]:
    """
    The RawStaffs might have gaps and segments start and end differently on every staff line.
    This function resamples the staffs so for every point of the staff we know the y positions
    of all staff lines. In the end this makes the staffs easier to use in the rest of
    the analysis.
    """
    result = []
    for staff in staffs:
        anchors_left_to_right = sorted(staff.anchors, key=lambda a: a.symbol.center[0])
        staff_density = 10
        start = (staff.min_x // staff_density) * staff_density
        stop = (staff.max_x // staff_density + 1) * staff_density

        grid: list[StaffPoint] = []
        x = start
        for i, anchor in enumerate(anchors_left_to_right):
            # to_left = x region from current x to current anchor
            to_left = range(int(x), int(anchor.symbol.center[0]), staff_density)

            # to_right = x region from anchor to halfway to next anchor (or to end)
            if i < len(anchors_left_to_right) - 1:
                to_right = range(
                    int(anchor.symbol.center[0]),
                    int(
                        (
                            anchor.symbol.center[0]
                            + anchors_left_to_right[i + 1].symbol.center[0]
                        )
                        / 2
                    ),
                    staff_density,
                )
            else:
                to_right = range(int(anchor.symbol.center[0]), int(stop), staff_density)

            x = to_right.stop

            # goes through each anchor, processing the left and right regions of each
            grid.extend(
                reversed(list(resample_staff_segment(anchor, staff, reversed(to_left))))
            )
            grid.extend(resample_staff_segment(anchor, staff, to_right))

        result.append(Staff(grid))

    return result


def filter_edge_of_vision(
    staffs: list[Staff], image_shape: tuple[int, ...]
) -> list[Staff]:
    """
    Removes staffs which begin at the right edge or at the lower edge of the image,
    as these are very likely incomplete staffs.
    """
    h, w = image_shape[0], image_shape[1]

    result = []
    for staff in staffs:
        starts_at_right_edge = staff.min_x > 0.90 * w
        starts_at_bottom_edge = staff.min_y > 0.95 * h
        ends_at_left_edge = staff.max_x < 0.20 * w
        if not (starts_at_right_edge or starts_at_bottom_edge or ends_at_left_edge):
            result.append(staff)
    return result
