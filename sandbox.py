import logging
import math
import json
import cv2
import numpy as np
import scipy.ndimage
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
from PIL import Image
from app.utils.download import download_models, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="(%(name)s:%(lineno)s) - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_PATH = "test_imgs/img1.JPG"

UNET_PATH = str(MODELS_DIR.joinpath("unet_91-df68794a7f3420b749780deb1eba938911b3d0d3"))
SEGNET_PATH = str(
    MODELS_DIR.joinpath("segnet_89-f8076e6ee78bf998e291a56647477de80aa19f64")
)


# download models
download_models()


# replace extension
def replace_extension(path: str, new_extension: str) -> str:
    return Path.cwd().joinpath(path).stem + new_extension


xml_path = replace_extension(IMAGE_PATH, ".musicxml")


########################################
# GENERAL UTILS
########################################
NDArray = np.ndarray[Any, Any]


########################################
# TESTING UTILS
########################################
def write_debug_image(
    image,
    name: str,
    bbox: Optional[tuple[int, int, int, int]] = None,
    binary_map: Optional[NDArray] = None,
):
    Path.cwd().joinpath("debug_images").mkdir(exist_ok=True)
    filepath = f"debug_images/{name}"
    if binary_map is not None:
        status = cv2.imwrite(filepath, binary_map * 255)
        if status:
            logger.info("image saved")
        else:
            raise Exception("image error")
    else:
        vis = image.copy()
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        status = cv2.imwrite(filepath, vis)
        if status:
            logger.info("image saved")
        else:
            raise Exception("image error")


########################################
# IMAGE PREPROCESSING UTILS
########################################
### CV2 UTILS
def get_ndarray_dims(image: NDArray) -> tuple[int, int]:
    """Returns the NDArray's height and width."""
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
            block_grid[i, j] = get_dominant_color(block, default=default_bg_color)  # type: ignore

    # Smooth the grid using blur
    blurred_grid = cv2.blur(block_grid, (3, 3))

    # Normalize brightness (lighten darker areas)
    WHITE = 255
    non_white = blurred_grid < WHITE
    max_brightness = int(np.max(blurred_grid[non_white]))
    blurred_grid[non_white] += WHITE - max_brightness

    # Resize blurred grid to image size
    background_image = cv2.resize(
        blurred_grid, (image_w, image_h), interpolation=cv2.INTER_LINEAR
    )

    # Normalize image by dividing by the created background image
    normalized = cv2.divide(image, background_image, scale=WHITE)

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


########################################
# MODEL INFERENCE UTILS
########################################
@dataclass
class SymbolMaps:
    staff: NDArray
    symbols: NDArray
    stems_rests: NDArray
    notehead: NDArray
    clefs_keys: NDArray


@dataclass
class SymbolMapsWithImages(SymbolMaps):
    original: NDArray
    preprocessed: NDArray


class InferenceModel:
    """Copied from homr/oemer"""

    def __init__(self, model_path: str) -> None:
        model, metadata = self._load_model(model_path)
        self.model = model
        self.input_shape = metadata["input_shape"]
        self.output_shape = metadata["output_shape"]

    def _load_model(self, model_path: str) -> tuple[Any, dict[str, Any]]:
        """Load model and metadata"""
        import tensorflow as tf

        model = tf.saved_model.load(model_path)
        with open(Path(model_path).joinpath("meta.json")) as f:
            metadata = json.loads(f.read())
        return model, metadata

    def inference(
        self,
        image: NDArray,
        step_size: int = 128,
        batch_size: int = 16,
        manual_th: Any | None = None,
    ) -> tuple[NDArray, NDArray]:

        # Collect data
        # Tricky workaround to avoid random mystery transpose when loading with 'Image'.
        image_rgb = Image.fromarray(image).convert("RGB")
        image = np.array(image_rgb)
        win_size = self.input_shape[1]
        data = []
        for y in range(0, image.shape[0], step_size):
            if y + win_size > image.shape[0]:
                y = image.shape[0] - win_size
            for x in range(0, image.shape[1], step_size):
                if x + win_size > image.shape[1]:
                    x = image.shape[1] - win_size
                hop = image[y : y + win_size, x : x + win_size]
                data.append(hop)

        # Predict
        pred = []
        for idx in range(0, len(data), batch_size):
            batch = np.array(data[idx : idx + batch_size])
            out = self.model.serve(batch)
            pred.append(out)

        # Merge prediction patches
        output_shape = image.shape[:2] + (self.output_shape[-1],)
        out = np.zeros(output_shape, dtype=np.float32)
        mask = np.zeros(output_shape, dtype=np.float32)
        hop_idx = 0
        for y in range(0, image.shape[0], step_size):
            if y + win_size > image.shape[0]:
                y = image.shape[0] - win_size
            for x in range(0, image.shape[1], step_size):
                if x + win_size > image.shape[1]:
                    x = image.shape[1] - win_size
                batch_idx = hop_idx // batch_size
                remainder = hop_idx % batch_size
                hop = pred[batch_idx][remainder]
                out[y : y + win_size, x : x + win_size] += hop
                mask[y : y + win_size, x : x + win_size] += 1
                hop_idx += 1

        out /= mask
        if manual_th is None:
            class_map = np.argmax(out, axis=-1)
        else:
            if len(manual_th) != output_shape[-1] - 1:
                raise ValueError(f"{manual_th}, {output_shape[-1]}")
            class_map = np.zeros(out.shape[:2] + (len(manual_th),))
            for idx, th in enumerate(manual_th):
                class_map[..., idx] = np.where(out[..., idx + 1] > th, 1, 0)

        return class_map, out


cached_segmentation: dict[str, Any] = {}


def run_segmentation_inference(
    model_path: str,
    image: NDArray,
    step_size: int = 128,
    batch_size: int = 16,
    manual_th: Optional[Any] = None,
) -> tuple[NDArray, NDArray]:
    if model_path not in cached_segmentation:
        model = InferenceModel(model_path)
        cached_segmentation[model_path] = model
    else:
        model = cached_segmentation[model_path]
    return model.inference(image, step_size, batch_size, manual_th)


def generate_segmentation_preds(
    original: NDArray, preprocessed: NDArray
) -> SymbolMapsWithImages:
    """
    Runs the segmentation models on the image and returns binary maps of each symbol type.
    """
    logger.info("Extracting staffline and symbols")
    staff_symbols_map, _ = run_segmentation_inference(UNET_PATH, preprocessed)
    staff_id = 1
    symbol_id = 2
    staff = np.where(staff_symbols_map == staff_id, 1, 0)
    symbols = np.where(staff_symbols_map == symbol_id, 1, 0)

    logger.info("Extracting symbol types")
    categorized, _ = run_segmentation_inference(SEGNET_PATH, preprocessed)
    stems_rests_id = 1
    notehead_id = 2
    clefs_keys_id = 3
    stems_rests = np.where(categorized == stems_rests_id, 1, 0)
    notehead = np.where(categorized == notehead_id, 1, 0)
    clefs_keys = np.where(categorized == clefs_keys_id, 1, 0)

    staff_map_h, staff_map_w = get_ndarray_dims(staff)
    original = cv2.resize(original, (staff_map_w, staff_map_h))
    preprocessed = cv2.resize(preprocessed, (staff_map_w, staff_map_h))
    return SymbolMapsWithImages(
        original=original,
        preprocessed=preprocessed,
        staff=staff.astype(np.uint8),
        symbols=symbols.astype(np.uint8),
        stems_rests=stems_rests.astype(np.uint8),
        notehead=notehead.astype(np.uint8),
        clefs_keys=clefs_keys.astype(np.uint8),
    )


########################################
# IMAGE POSTPROCESSING UTILS
########################################
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
    NOISE_LIMIT = 50

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
            any_neighbor_above_limit = np.any(np.array(neighbors) > NOISE_LIMIT)

            if noise_level > NOISE_LIMIT and any_neighbor_above_limit:
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


########################################
# BOUNDING BOX UTILS
########################################
@dataclass
class 


def create_notehead_bboxes(img: NDArray, min_size: Optional[tuple[int, int]] =(4, 4)) -> list[]


########################################

# detect staffs in image
logger.info("Detecting staffs")
## loading/preprocessing segmentation predictions
logger.info("Loading segmentation")
### image preprocessing
image = cv2.imread(IMAGE_PATH)
image = autocrop(image)
image = resize_image(image)
preprocessed, _ = color_adjust(image)

### model inference
predictions = generate_segmentation_preds(image, preprocessed)

### image postprocessing
predictions = filter_segmentation_preds(predictions)
predictions.staff = make_lines_stronger(predictions.staff)
logger.info("Loaded segmentation")

## predicting symbols
logger.info("Creating bounding boxes for noteheads")

write_debug_image(image, "staff.png", binary_map=predictions.staff)
write_debug_image(image, "symbols.png", binary_map=predictions.symbols)
write_debug_image(image, "stems_rests.png", binary_map=predictions.stems_rests)
write_debug_image(image, "notehead.png", binary_map=predictions.notehead)
write_debug_image(image, "clefs_keys.png", binary_map=predictions.clefs_keys)
