import io
import logging
import math
import cv2
import numpy as np
import scipy.ndimage
from typing import Optional
from PIL import Image

from worker.utils.constants import NDArray, get_ndarray_dims


logger = logging.getLogger(__name__)


### SIZE UTILS
def get_target_size(image: Image.Image) -> tuple[int, int]:
    """
    Calculate target size by optimizing number of pixels to 3M~4.35M.
    """
    min_pixels = 3.00 * 1000 * 1000
    max_pixels = 4.35 * 1000 * 1000

    w, h = image.size
    pixels = w * h
    if min_pixels <= pixels <= max_pixels:
        return w, h

    lower_bound = min_pixels / pixels
    upper_bound = max_pixels / pixels
    avg_bound = (lower_bound + upper_bound) / 2
    scale_factor = pow(avg_bound, 0.5)

    target_w, target_h = round(scale_factor * w), round(scale_factor * h)
    return target_w, target_h


def resize_image(image_arr: NDArray) -> NDArray:
    orig_h, orig_w = get_ndarray_dims(image_arr)

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
        [gray],
        [0],
        None,
        [256],
        [0, 256],  # freqs of each intensity value
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
    img_h, img_w = get_ndarray_dims(img)
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
) -> int:
    if grayscale.dtype != np.uint8:
        raise TypeError("Image must be of dtype uint8")

    # Create a boolean mask for values in the range [min_val, max_val]
    mask = (grayscale >= min_val) & (grayscale <= max_val)

    # Apply mask to grayscale image
    masked_grayscale = grayscale[mask]
    if masked_grayscale.size == 0:
        return 0 if default is None else default

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
    image_h, image_w = get_ndarray_dims(image)
    x_range = range(0, image_w, block_size)
    y_range = range(0, image_h, block_size)

    # Stores dominant color of each block
    block_grid = np.zeros(
        [math.ceil(image_h / block_size), math.ceil(image_w / block_size)],
        dtype=np.uint8,
    )
    default_bg_color = get_dominant_color(image)
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            pixel_coords = (y, x)
            block_coords = get_block_coords(image.shape, pixel_coords, block_size)
            block = image[block_coords]
            block_grid[i, j] = get_dominant_color(block, default=default_bg_color)

    # Smooth the grid using blur
    blurred_grid = cv2.blur(block_grid, (3, 3))

    # Normalize brightness (lighten darker areas)
    white = 255
    non_white = blurred_grid < white
    max_brightness = int(np.max(blurred_grid[non_white]))
    blurred_grid[non_white] += white - max_brightness

    # Resize blurred grid to image size
    background_image = cv2.resize(
        blurred_grid, (image_w, image_h), interpolation=cv2.INTER_LINEAR
    )

    # Normalize image by dividing by the created background image
    normalized = cv2.divide(image, background_image, scale=white)

    return normalized, background_image


def apply_contrast(image: NDArray) -> NDArray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def color_adjust(image: NDArray, block_size: int = 40) -> tuple[NDArray, NDArray]:
    """
    Reduce the effect of uneven lighting on the image by dividing the image by its interpolated
    background.
    """
    try:
        logger.info("Adjusting color of image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image, background = normalize_background(image, block_size)
        return cv2.cvtColor(apply_contrast(image), cv2.COLOR_GRAY2BGR), background
    except Exception as e:
        logger.error(f"Error while adjusting color of image: {str(e)}")
        return image, image


def preprocess_image_from_bytes(image_bytes: bytes) -> bytes:
    # image = cv2.imread(image_path)

    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    image = autocrop(image)
    image = resize_image(image)

    buffer = io.BytesIO()
    np.save(buffer, image)
    buffer.seek(0)
    serialized = buffer.read()

    return serialized
