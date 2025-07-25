import logging
from worker.inference.inference import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="(%(name)s:%(lineno)s) - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


image_paths = ["worker/test_imgs/img1.JPG", "worker/test_imgs/img2.JPG"]
run_inference(image_paths)
