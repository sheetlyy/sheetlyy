from typing import Annotated

from litestar import get, post, Request
from litestar.response import Template
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.plugins.htmx import HTMXTemplate
# from worker.inference.inference import run_inference


@get("/", sync_to_thread=False)
def index() -> Template:
    # run_inference(["worker/test_imgs/img1.JPG", "worker/test_imgs/img2.JPG"])

    context = {"index": "1"}
    return HTMXTemplate(template_name="base.html", context=context)


@get("/add-file", sync_to_thread=False)
def add_file_input(index: int) -> Template:
    context = {"index": index + 1}
    return HTMXTemplate(template_name="fragments/file_input.html", context=context)


@post("/upload")
async def handle_file_uploads(
    request: Annotated[Request, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Template:
    form = await request.form()

    ordered_files: list[tuple[int, UploadFile]] = sorted(
        [(int(input_name), form[input_name]) for input_name in form],
        key=lambda x: x[0],
    )
    files = [(file.filename, await file.read()) for _, file in ordered_files]

    context = {"filenames": [file[0] for file in files]}
    return HTMXTemplate(template_name="fragments/uploaded.html", context=context)


@get("/books/{book_id:int}", sync_to_thread=False)
def get_book(book_id: int) -> dict[str, int]:
    return {"book_id": book_id}
