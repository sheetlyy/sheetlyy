from typing import Annotated

from litestar import get, post, Request
from litestar.response import Template
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.plugins.htmx import HTMXTemplate

from worker.utils.image_preprocessing import preprocess_image_from_bytes
from worker.inference.inference import run_inference


@get("/", sync_to_thread=False)
def index() -> Template:
    context = {"index": "1"}
    return HTMXTemplate(template_name="base.html", context=context)


@get("/add-file", sync_to_thread=False)
def add_file_input(index: int) -> Template:
    is_max_reached = index + 1 >= 10
    context = {"index": index + 1, "is_max_reached": is_max_reached}
    return HTMXTemplate(template_name="fragments/file_input.html", context=context)


@post("/upload")
async def handle_file_uploads(
    request: Annotated[Request, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Template:
    # run_inference(["worker/test_imgs/img1.JPG", "worker/test_imgs/img2.JPG"])

    form = await request.form()

    ordered_files: list[UploadFile] = [
        form[file_idx] for file_idx in sorted(form, key=int)
    ]

    image_arrs = [
        preprocess_image_from_bytes(await file.read()) for file in ordered_files
    ]
    # run_inference(image_arrs)

    context = {"filenames": [arr.size for arr in image_arrs]}
    return HTMXTemplate(template_name="fragments/uploaded.html", context=context)


@get("/books/{book_id:int}", sync_to_thread=False)
def get_book(book_id: int) -> dict[str, int]:
    return {"book_id": book_id}
