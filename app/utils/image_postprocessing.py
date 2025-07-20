import logging
import cv2
import numpy as np
from typing import Optional

from app.utils.constants import NDArray, get_ndarray_dims
from app.segmentation.inference import SymbolMapsWithImages

logger = logging.getLogger(__name__)


### NOISE FILTERING UTILS
def estimate_noise(tile: NDArray) -> int:
    """Estimate average noise level in tile using a Laplacian kernel."""
    tile_h, tile_w = get_ndarray_dims(tile)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    noise_level = np.sum(
        np.sum(np.absolute(cv2.filter2D(tile, cv2.CV_64F, kernel)))
    ) / (tile_h * tile_w)
    return noise_level


def get_neighbors(grid: NDArray, tile_y: int, tile_x: int) -> list[int]:
    """Returns the noise levels of a tile's adjacent neighbors."""
    max_h, max_w = get_ndarray_dims(grid)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    neighbors = [
        grid[tile_y + step_y, tile_x + step_x]
        for step_y, step_x in steps
        if 0 <= tile_y + step_y < max_h and 0 <= tile_x + step_x < max_w
    ]

    return neighbors


def create_noise_mask(grayscale: NDArray) -> Optional[NDArray]:
    """
    Divides image into 20x20 grid, calculates noise per grid tile, and
    filters out tiles that are too noisy.
    """
    noise_limit = 50

    image_h, image_w = get_ndarray_dims(grayscale)
    mask = np.zeros(grayscale.shape, dtype=np.uint8)
    tile_h, tile_w = image_h // 20, image_w // 20

    filtered, total = 0, 0

    # Create 20x20 grid with noise levels per grid tile
    grid = np.zeros(
        [int(np.ceil(image_h / tile_h)), int(np.ceil(image_w / tile_w))],
        dtype=np.uint8,
    )

    # First pass to store noise level per grid tile
    for i, y1 in enumerate(range(0, image_h, tile_h)):
        for j, x1 in enumerate(range(0, image_w, tile_w)):
            y2, x2 = y1 + tile_h, x1 + tile_w
            tile = grayscale[y1:y2, x1:x2]
            noise_level = estimate_noise(tile)
            grid[i, j] = noise_level

    # Second pass to analyze local noise level and determine whether to filter
    for i, y1 in enumerate(range(0, image_h, tile_h)):
        for j, x1 in enumerate(range(0, image_w, tile_w)):
            y2, x2 = y1 + tile_h, x1 + tile_w
            noise_level = grid[i, j]
            neighbors = get_neighbors(grid, i, j)
            any_neighbor_above_limit = np.any(np.array(neighbors) > noise_limit)

            if noise_level > noise_limit and any_neighbor_above_limit:
                filtered += 1
            else:
                mask[y1:y2, x1:x2] = 255

            total += 1

    # If <= 50% of image is filtered, return mask
    if filtered / total > 0.5:
        logger.info(
            f"Would filter more than 50% of image ({filtered}/{total} cells), skipping noise filtering"
        )
        return None
    elif filtered > 0:
        logger.info(f"Filtered {filtered} of {total} tiles")
        return mask
    logger.info("No tiles filtered")
    return None


def filter_segmentation_preds(
    predictions: SymbolMapsWithImages,
) -> SymbolMapsWithImages:
    logger.info("Performing noise filtering")
    mask = create_noise_mask(255 * predictions.staff)
    if mask is None:
        return predictions
    return SymbolMapsWithImages(
        original=cv2.bitwise_and(predictions.original, predictions.original, mask=mask),
        preprocessed=cv2.bitwise_and(
            predictions.preprocessed, predictions.preprocessed, mask=mask
        ),
        staff=cv2.bitwise_and(predictions.staff, predictions.staff, mask=mask),
        symbols=cv2.bitwise_and(predictions.symbols, predictions.symbols, mask=mask),
        stems_rests=cv2.bitwise_and(
            predictions.stems_rests, predictions.stems_rests, mask=mask
        ),
        notehead=cv2.bitwise_and(predictions.notehead, predictions.notehead, mask=mask),
        clefs_keys=cv2.bitwise_and(
            predictions.clefs_keys, predictions.clefs_keys, mask=mask
        ),
    )


### LINE ENHANCEMENT UTILS
def make_lines_stronger(img: NDArray, kernel_size: tuple[int, int] = (1, 2)) -> NDArray:
    logger.info("Strengthening staff lines")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img = cv2.dilate(img.astype(np.uint8), kernel)
    img = cv2.threshold(img, 0.1, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    return img
