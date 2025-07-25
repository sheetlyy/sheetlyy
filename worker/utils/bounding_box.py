import logging
import math
import cv2
import cv2.typing as cvt
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from collections import defaultdict

from worker.utils.constants import NDArray, max_line_gap_size


logger = logging.getLogger(__name__)


def do_polygons_overlap(poly1: cvt.MatLike, poly2: cvt.MatLike) -> bool:
    # check if any point of one ellipse is inside other ellipse
    for point in poly1:
        if cv2.pointPolygonTest(poly2, (float(point[0]), float(point[1])), False) >= 0:
            return True
    for point in poly2:
        if cv2.pointPolygonTest(poly1, (float(point[0]), float(point[1])), False) >= 0:
            return True
    return False


class DebugDrawable(ABC):
    @abstractmethod
    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)
    ) -> None:
        pass


class Polygon(DebugDrawable):
    def __init__(self, polygon: Any):
        self.polygon = polygon


class BoundingBox(Polygon):
    """
    A bounding box in the format of (x1, y1, x2, y2)
    """

    def __init__(self, box: cvt.Rect, contours: cvt.MatLike):
        self.box = box
        self.contours = contours

        self.center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.size = (box[2] - box[0], box[3] - box[1])
        self.rotated_box = (self.center, self.size, 0)
        super().__init__(cv2.boxPoints(self.rotated_box).astype(np.int64))

    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)
    ) -> None:
        x1, y1, x2, y2 = self.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


class AngledBoundingBox(Polygon):
    def __init__(
        self,
        box: cvt.RotatedRect,
        contours: cvt.MatLike,
        polygon: Any,
    ):
        super().__init__(polygon)
        self.contours = contours

        angle = box[2]
        self.box: cvt.RotatedRect  # RotatedRect: ((cx, cy), (w, h), angle)
        if angle > 135 or angle < -135:
            angle += -180 if angle > 135 else 180
            size = (box[1][0], box[1][1])
        elif angle > 45 or angle < -45:
            angle += -90 if angle > 45 else 90
            size = (box[1][1], box[1][0])
        else:
            size = (box[1][0], box[1][1])
        self.box = ((box[0][0], box[0][1]), size, angle)

        self.center = self.box[0]
        self.size = self.box[1]
        self.angle = self.box[2]

        self.top_left, self.bottom_left, self.top_right, self.bottom_right = (
            self.calculate_corners()
        )

    def calculate_corners(self) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        half_size = np.array([self.size[0] / 2, self.size[1] / 2])
        half_w, half_h = half_size

        top_left = self.center - half_size
        bottom_left = self.center + np.array([-half_w, half_h])
        top_right = self.center + np.array([half_w, -half_h])
        bottom_right = self.center + half_size

        return (
            tuple(top_left),
            tuple(bottom_left),
            tuple(top_right),
            tuple(bottom_right),
        )

    def is_overlapping(self, other: Polygon) -> bool:
        if not self._can_shapes_possibly_touch(other):
            return False
        return do_polygons_overlap(self.polygon, other.polygon)

    def is_overlapping_with_any(self, others: Sequence["AngledBoundingBox"]) -> bool:
        return any(self.is_overlapping(other) for other in others)

    def _can_shapes_possibly_touch(self, other: Polygon) -> bool:
        """
        A fast check if the two shapes can possibly touch. If this returns False,
        the two shapes do not touch.
        If this returns True, the two shapes might touch and further checks are necessary.
        """

        # get centers and major axes of the rectangles
        center1, axes1, _ = self.box
        center2: Sequence[float]
        axes2: Sequence[float]
        if isinstance(other, BoundingBox):
            # rotated_box: tuple[tuple[float, float], tuple[int, int], Literal[0]]
            center2, axes2, _ = other.rotated_box
        elif isinstance(other, AngledBoundingBox):
            # box: tuple[tuple[float, float], tuple[int, int], float]
            center2, axes2, _ = other.box
        else:
            raise ValueError(f"Unknown type {type(other)}")
        major_axis1 = max(axes1)
        major_axis2 = max(axes2)

        # calculate distance between centers
        distance = (
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        ) ** 0.5

        # if distance > sum of major axes, there is no overlap
        if distance > major_axis1 + major_axis2:
            return False
        return True

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, AngledBoundingBox):
            return self.box == __value.box
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.box)

    def __str__(self) -> str:
        return str(self.box)

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)
    ) -> None:
        pass


class RotatedBoundingBox(AngledBoundingBox):
    def __init__(self, box: cvt.RotatedRect, contours: cvt.MatLike):
        super().__init__(box, contours, cv2.boxPoints(box).astype(np.int64))

    def is_intersecting(self, other: "RotatedBoundingBox") -> bool:
        if not self._can_shapes_possibly_touch(other):
            return False
        return (
            cv2.rotatedRectangleIntersection(self.box, other.box)[0]
            != cv2.INTERSECT_NONE
        )

    def is_overlapping_extrapolated(
        self, other: "RotatedBoundingBox", unit_size: float
    ) -> bool:
        """
        Check if two horizontal staff line fragments are close enough (in space and slope)
        to be considered overlapping or continuous, even if there is a small gap.
        """
        if self.center[0] > other.center[0]:
            left, right = other, self
        else:
            left, right = self, other
        center: float = float(np.mean([left.center[0], right.center[0]]))
        tolerance = unit_size / 3
        max_gap = max_line_gap_size(unit_size)

        left_gap = center - (left.center[0] + left.size[0] // 2)
        right_gap = (right.center[0] - right.size[0] // 2) - center
        if left_gap > max_gap or right_gap > max_gap:
            return False

        vertical_diff = abs(
            left.get_center_extrapolated(center) - right.get_center_extrapolated(center)
        )
        if vertical_diff > tolerance:
            return False

        return True

    def make_box_thicker(self, thickness: int) -> "RotatedBoundingBox":
        if thickness <= 0:
            return self
        return RotatedBoundingBox(
            (
                self.center,
                (self.size[0] + thickness, self.size[1] + thickness),
                self.angle,
            ),
            self.contours,
        )

    def move_x_horizontal_by(self, x_delta: int) -> "RotatedBoundingBox":
        new_x = self.center[0] + x_delta
        return RotatedBoundingBox(
            ((new_x, self.center[1]), self.size, self.angle),
            self.contours,
        )

    def make_taller_by(self, thickness: int) -> "RotatedBoundingBox":
        return RotatedBoundingBox(
            (self.center, (self.size[0], self.size[1] + thickness), self.angle),
            self.contours,
        )

    def get_center_extrapolated(self, x: float) -> float:
        """
        Returns the Y position at a given X,
        based on the angle of the rotated bounding box.
        """
        return (x - self.center[0]) * np.tan(self.angle / 180 * np.pi) + self.center[1]

    def to_bounding_box(self) -> BoundingBox:
        return BoundingBox(
            (
                int(self.top_left[0]),
                int(self.top_left[1]),
                int(self.bottom_right[0]),
                int(self.bottom_right[1]),
            ),
            self.contours,
        )

    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)
    ) -> None:
        box = cv2.boxPoints(self.box).astype(np.int64)
        cv2.drawContours(img, [box], 0, color, 2)


class BoundingEllipse(AngledBoundingBox):
    def __init__(
        self,
        box: cvt.RotatedRect,
        contours: cvt.MatLike,
    ):
        super().__init__(
            box,
            contours,
            cv2.ellipse2Poly(
                (int(box[0][0]), int(box[0][1])),
                (int(box[1][0] / 2), int(box[1][1] / 2)),
                int(box[2]),
                0,
                360,
                1,
            ),
        )

    def make_box_thicker(self, thickness: int) -> "BoundingEllipse":
        return BoundingEllipse(
            (
                self.center,
                (self.size[0] + thickness, self.size[1] + thickness),
                self.angle,
            ),
            self.contours,
        )

    def draw_onto_image(
        self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)
    ) -> None:
        cv2.ellipse(img, self.box, color=color, thickness=2)


@dataclass
class SymbolBoundingBoxes:
    noteheads: list[BoundingEllipse]
    staff_fragments: list[RotatedBoundingBox]
    clefs_keys: list[RotatedBoundingBox]
    accidentals: list[RotatedBoundingBox]
    stems_rests: list[RotatedBoundingBox]
    bar_lines: list[RotatedBoundingBox]


class UnionFind:
    def __init__(self, n: int):
        self.parent: list[int] = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # union by rank to keep tree flat
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


def has_box_valid_size(box: cvt.RotatedRect) -> bool:
    box_w, box_h = box[1][0], box[1][1]
    return not math.isnan(box_w) and not math.isnan(box_h) and box_w > 0 and box_h > 0


def merge_overlaying_bboxes(
    boxes: Sequence[AngledBoundingBox],
) -> list[list[AngledBoundingBox]]:
    n = len(boxes)
    uf = UnionFind(n)

    # try to find overlaps and union groups that overlap
    for i in range(n):
        for j in range(i + 1, n):
            if boxes[i].is_overlapping(boxes[j]):
                uf.union(i, j)

    # create merged groups based on the union-find results
    merged_groups: dict[int, list[AngledBoundingBox]] = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        merged_groups[root].append(boxes[i])

    return list(merged_groups.values())


def create_bounding_ellipses(
    img: NDArray,
    min_size: Optional[tuple[int, int]] = (4, 4),
) -> list[BoundingEllipse]:
    """
    Fits and filters ellipses, merges overlapping ones into groups, and fits
    one bounding ellipse per group.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for contour in contours:
        min_length_to_fit_ellipse = 5
        if len(contour) < min_length_to_fit_ellipse:
            continue

        fitbox = cv2.fitEllipse(contour)
        if not has_box_valid_size(fitbox):
            continue

        ellipse = BoundingEllipse(fitbox, contour)
        if min_size and (
            ellipse.size[0] < min_size[0] or ellipse.size[1] < min_size[1]
        ):
            continue

        ellipses.append(ellipse)

    # merge overlapping ellipses into groups and fit one ellipse per group
    groups = merge_overlaying_bboxes(ellipses)
    result = []
    for group in groups:
        complete_contour = np.concatenate([e.contours for e in group])
        box = cv2.minAreaRect(complete_contour)
        result.append(BoundingEllipse(box, complete_contour))

    return result


def create_rotated_bboxes(
    img: NDArray,
    skip_merging: bool = False,
    min_size: Optional[tuple[int, int]] = None,
    max_size: Optional[tuple[int, int]] = None,
) -> list[RotatedBoundingBox]:
    """
    Fits and filters boxes, merges overlapping ones into groups, and fits
    one rotated bounding box per group.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[RotatedBoundingBox] = []
    for contour in contours:
        fitbox = cv2.minAreaRect(contour)
        if not has_box_valid_size(fitbox):
            continue

        box = RotatedBoundingBox(fitbox, contour)
        box_w, box_h = box.size

        if min_size and (box_w < min_size[0] or box_h < min_size[1]):
            continue
        if max_size:
            if (max_size[0] > 0 and box_w > max_size[0]) or (
                max_size[1] > 0 and box_h > max_size[1]
            ):
                continue

        boxes.append(box)

    if skip_merging:
        return boxes

    # merge overlapping boxes into groups and fit one box per group
    groups = merge_overlaying_bboxes(boxes)
    result = []
    for group in groups:
        complete_contour = np.concatenate([box.contours for box in group])
        box = cv2.minAreaRect(complete_contour)
        result.append(RotatedBoundingBox(box, complete_contour))

    return result
