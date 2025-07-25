import logging
import cv2
from typing import Optional, Sequence
from pathlib import Path

from worker.utils.constants import NDArray
from worker.utils.bounding_box import DebugDrawable


logger = logging.getLogger(__name__)


WRITE_DEBUG_IMAGE = False


def write_debug_image(
    image,
    name: str,
    binary_map: Optional[NDArray] = None,
    drawables: Optional[Sequence[DebugDrawable]] = None,
):
    if not WRITE_DEBUG_IMAGE:
        return None

    Path.cwd().joinpath("debug_imgs").mkdir(exist_ok=True)
    filepath = f"debug_imgs/{name}"
    if binary_map is not None:
        status = cv2.imwrite(filepath, binary_map * 255)
    elif drawables:
        vis = image.copy()
        for d in drawables:
            d.draw_onto_image(vis)
        status = cv2.imwrite(filepath, vis)

    if status:
        logger.info(f"image saved: {name}")
    else:
        raise Exception(f"image error: {name}")
