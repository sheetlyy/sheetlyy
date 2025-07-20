import numpy as np
from typing import Any


NDArray = np.ndarray[Any, Any]

NUMBER_OF_LINES_ON_A_STAFF = 5
# We use ³ as triplet indicator as it's not a valid duration name
# or note name and thus we have no risk of confusion
TRIPLET_SYMBOL = "³"
DURATION_OF_QUARTER = 16


def max_line_gap_size(unit_size: float) -> float:
    return 5 * unit_size


### CV2 UTILS
def get_ndarray_dims(image: NDArray) -> tuple[int, int]:
    """Returns the NDArray's height and width."""
    image_h, image_w = image.shape[0], image.shape[1]
    return image_h, image_w
