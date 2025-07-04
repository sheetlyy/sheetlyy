import logging
import math
import cv2
import numpy as np
import scipy.ndimage
from typing import Any, Optional
from pathlib import Path
from PIL import Image
from app.utils.models import download_models

logging.basicConfig(
    level=logging.DEBUG,
    format="(%(name)s:%(lineno)s) - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_PATH = "test_images/img1.JPG"

# download models
download_models()


# replace extension
def replace_extension(path: str, new_extension: str) -> str:
    return Path.cwd().joinpath(path).stem + new_extension


xml_path = replace_extension(IMAGE_PATH, ".musicxml")


########################################
# TESTING UTILS
########################################
# vis = img.copy()
# cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
# status = cv2.imwrite("vis_bbox.jpg", vis)
# if status:
#     logger.info("image saved")
# else:
#     raise Exception("image error")


########################################
# IMAGE MANIPULATION UTILS
########################################
### CV2 UTILS
NDArray = np.ndarray[Any, Any]


def get_image_dims(image: NDArray) -> tuple[int, int]:
    """Returns the image's height and width."""
    image_h, image_w = image.shape[0], image.shape[1]
    return image_h, image_w


### SIZE UTILS
def get_target_size(image: Image.Image) -> tuple[int, int]:
    """
    Calculate target size by optimizing number of pixels to 3M~4.35M.
    """
    MIN_PIXELS = 3.00 * 1000 * 1000
    MAX_PIXELS = 4.35 * 1000 * 1000

    w, h = image.size
    pixels = w * h
    if MIN_PIXELS <= pixels <= MAX_PIXELS:
        return w, h

    lower_bound = MIN_PIXELS / pixels
    upper_bound = MAX_PIXELS / pixels
    avg_bound = (lower_bound + upper_bound) / 2
    scale_factor = pow(avg_bound, 0.5)

    target_w, target_h = round(scale_factor * w), round(scale_factor * h)
    return target_w, target_h


def resize_image(image_arr: NDArray) -> NDArray:
    orig_h, orig_w = get_image_dims(image_arr)

    image = Image.fromarray(image_arr)
    target_w, target_h = get_target_size(image)
    if target_w == orig_w and target_h == orig_h:
        logger.info(f"Keeping original size of {target_w} x {target_h}")
        return image_arr

    logger.info(f"Resizing image from {orig_w} x {orig_h} to {target_w} x {target_h}")
    return np.array(image.resize((target_w, target_h)))


### CROPPING UTILS
def autocrop(img: NDArray) -> NDArray:
    """
    Find the largest contour on the image, which is expected to be the paper of sheet music
    and extracts it from the image. If no contour is found, then the image is assumed to be
    a full page view of sheet music and is returned as is.
    """
    # convert to grayscale & find most frequent intensity value
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist(
        [gray], [0], None, [256], [0, 256]  # freqs of each intensity value
    )
    dominant_intensity_val = max(enumerate(hist), key=lambda x: x[1])[0]

    # threshold (convert to binary image)
    thresh = cv2.threshold(gray, dominant_intensity_val - 30, 255, cv2.THRESH_BINARY)[1]

    # apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((9, 9), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel)

    # get largest contour (paper)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.info("No contours found, skipping crop")
        return img
    big_contour = max(contours, key=cv2.contourArea)

    # get bounding box
    x, y, w, h = cv2.boundingRect(big_contour)
    img_h, img_w = get_image_dims(img)
    # if contour isn't large enough, assume image doesn't have page borders
    is_full_page_view = w < img_w * 0.25 or h < img_h * 0.25
    if is_full_page_view:
        logger.info("Image likely contains full page view, skipping crop")
        return img

    # crop result
    logger.info("Cropping image")
    cropped = img[y : y + h, x : x + w]
    return cropped


### COLOR UTILS
def get_dominant_color(
    grayscale: NDArray,
    min_val: int = 150,
    max_val: int = 254,
    default: Optional[int] = None,
) -> Optional[int]:
    if grayscale.dtype != np.uint8:
        raise TypeError("Image must be of dtype uint8")

    # Create a boolean mask for values in the range [min_val, max_val]
    mask = (grayscale >= min_val) & (grayscale <= max_val)

    # Apply mask to grayscale image
    masked_grayscale = grayscale[mask]
    if masked_grayscale.size == 0:
        return default

    bins = np.bincount(masked_grayscale.flatten())
    center_of_mass = scipy.ndimage.center_of_mass(bins)[0]

    return int(center_of_mass)  # type: ignore


def get_block_coords(
    image_shape: tuple[int, ...],
    pixel_coords: tuple[int, int],
    block_size: int,
) -> tuple[NDArray, ...]:
    """
    Creates a grid of indices for a block of pixels around a given pixel and returns the
    block's coordinates.
    """
    half_block = block_size // 2
    image_h, image_w = image_shape
    pixel_y, pixel_x = pixel_coords
    y = np.arange(max(0, pixel_y - half_block), min(image_h, pixel_y + half_block))
    x = np.arange(max(0, pixel_x - half_block), min(image_w, pixel_x + half_block))
    return np.ix_(y, x)


def normalize_background(image: NDArray, block_size: int) -> tuple[NDArray, NDArray]:
    """
    Divides the image into blocks of size block_size and calculates
    the dominant color of each block. The dominant color is then
    used to create a background image, which is then used to divide the
    original image. The result is an image with a more uniform background.
    """
    image_h, image_w = get_image_dims(image)
    x_range = range(0, image_w, block_size)
    y_range = range(0, image_h, block_size)
    block_grid = np.zeros(
        [math.ceil(image_h / block_size), math.ceil(image_w / block_size)],
        dtype=np.uint8,
    )

    background_color = get_dominant_color(image)
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            pixel_coords = (y, x)
            block_coords = get_block_coords(image.shape, pixel_coords, block_size)
            block = image[block_coords]
            block_grid[i, j] = get_dominant_color(block, default=background_color)  # type: ignore

    background_blurred = cv2.blur()


def color_adjust(image: NDArray, block_size: int = 40) -> tuple[NDArray, NDArray]:
    """
    Reduce the effect of uneven lighting on the image by dividing the image by its interpolated
    background.
    """
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        pass


########################################

# detect staffs in image
# loading/preprocessing segmentation predictions
image = cv2.imread(IMAGE_PATH)
image = autocrop(image)
image = resize_image(image)
