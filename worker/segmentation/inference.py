import logging
import json
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
from PIL import Image

from worker.utils.constants import NDArray
from worker.utils.image_preprocessing import get_ndarray_dims
from worker.utils.download import MODELS_DIR


logger = logging.getLogger(__name__)

UNET_PATH = str(MODELS_DIR.joinpath("unet"))
SEGNET_PATH = str(MODELS_DIR.joinpath("segnet"))


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
