import logging
import numpy as np
from typing import Optional
from skimage import transform

from worker.utils.constants import NDArray
from worker.classes.models import Staff


logger = logging.getLogger(__name__)


class StaffDewarping:
    def __init__(self, tform: Optional[transform.PiecewiseAffineTransform]):
        self.tform = tform

    def dewarp(self, image: NDArray, fill_color: int = 1, order: int = 1) -> NDArray:
        if self.tform is None:
            return image
        return transform.warp(
            image,
            self.tform.inverse,
            output_shape=image.shape,
            mode="constant",
            order=order,
            cval=fill_color,
        )

    def dewarp_point(self, point: tuple[float, float]) -> tuple[float, float]:
        if self.tform is None:
            return point
        return self.tform(point)  # type: ignore


class FastPiecewiseAffineTransform(transform.PiecewiseAffineTransform):
    """
    From https://github.com/scikit-image/scikit-image/pull/6963/files
    """

    def __call__(self, coords):  # type: ignore
        coords = np.asarray(coords)

        simplex = self._tesselation.find_simplex(coords)  # type: ignore

        affines = np.stack([affine.params for affine in self.affines])[simplex]  # type: ignore

        points = np.c_[coords, np.ones((coords.shape[0], 1))]

        result = np.einsum("ikj,ij->ik", affines, points)
        result[simplex == -1, :] = -1
        result = result[:, :2]

        return result


def is_point_on_image(pts: tuple[int, int], image: NDArray) -> bool:
    x, y = pts
    h, w = image.shape[:2]
    margin = 10
    if x < margin or x > w - margin or y < margin or y > h - margin:
        return False
    return True


def calculate_dewarp_transformation(
    image: NDArray,
    source: list[list[tuple[int, int]]],
    destination: list[list[tuple[int, int]]],
    fast: bool = False,
) -> StaffDewarping:
    def add_image_edges_to_lines(
        lines: list[list[tuple[int, int]]],
    ) -> list[list[tuple[int, int]]]:
        lines.insert(0, [(0, 0), (0, image.shape[1])])
        lines.append([(image.shape[0], 0), (image.shape[0], image.shape[1])])
        return lines

    def add_first_and_last_point_to_every_line(
        lines: list[list[tuple[int, int]]],
    ) -> list[list[tuple[int, int]]]:
        for line in lines:
            line.insert(0, (0, line[0][1]))
            line.append((image.shape[1], line[-1][1]))
        return lines

    source = add_image_edges_to_lines(add_first_and_last_point_to_every_line(source))
    destination = add_image_edges_to_lines(
        add_first_and_last_point_to_every_line(destination)
    )

    # convert points to numpy arrays
    source_conc = np.concatenate(source)
    destination_conc = np.concatenate(destination)

    tform = (
        FastPiecewiseAffineTransform() if fast else transform.PiecewiseAffineTransform()
    )
    tform.estimate(source_conc, destination_conc)
    return StaffDewarping(tform)


def calculate_span_and_optimal_points(
    staff: Staff, image: NDArray
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    span_points: list[list[tuple[int, int]]] = []
    optimal_points: list[list[tuple[int, int]]] = []
    total_y_intervals = 6

    if int(image.shape[0] / total_y_intervals) == 0:
        return span_points, optimal_points

    first_y_offset = None
    for y in range(2, image.shape[0] - 2, int(image.shape[0] / total_y_intervals)):
        line_points: list[tuple[int, int]] = []
        for x in range(2, image.shape[1], 80):
            y_values = staff.get_at(x)
            if y_values is None:
                continue

            y_offset = y_values.y[2]
            if not first_y_offset:
                first_y_offset = y_offset
                y_delta = 0
            else:
                y_delta = int(y_offset - first_y_offset)

            point = (x, y + y_delta)
            if is_point_on_image(point, image):
                line_points.append(point)

        min_number_of_points = 2
        if len(line_points) > min_number_of_points:
            average_y = sum([p[1] for p in line_points]) / len(line_points)
            span_points.append(line_points)
            optimal_points.append([(p[0], int(average_y)) for p in line_points])

    return span_points, optimal_points


def dewarp_staff_image(image: NDArray, staff: Staff, index: int) -> StaffDewarping:
    try:
        span_points, optimal_points = calculate_span_and_optimal_points(staff, image)
        return calculate_dewarp_transformation(image, span_points, optimal_points)
    except Exception as e:
        logger.error(f"Dewarping failed for staff {index} with error: {e}")
    return StaffDewarping(None)
