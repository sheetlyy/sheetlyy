import uuid
import base64
from dataclasses import dataclass
from typing import Annotated

from litestar import post, Request
from litestar.response import Template
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.plugins.htmx import HTMXTemplate

from web.utils.clients import r, q
from web.utils.constants import REDIS_TIMEOUT
from worker.utils.image_preprocessing import preprocess_image
from worker.main import run_inference


def get_file_key(file_id: str) -> str:
    return f"file:{file_id}"


@dataclass
class FileMetadata:
    """
    File metadata used for rendering file thumbnails in the drag-and-drop UI.
    """

    file_id: str
    filename: str
    content_type: str
    thumbnail_b64: str


@post("/upload")
async def handle_file_uploads(
    data: Annotated[list[UploadFile], Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Template:
    """
    Accepts files from user, compresses them, stores them temporarily in Redis,
    and returns a view that allows the user to reorder files.
    """
    files_metadata: list[FileMetadata] = []
    for file in data:
        file_id = str(uuid.uuid4())
        file_data = await file.read()

        r.set(get_file_key(file_id), file_data, ex=REDIS_TIMEOUT)

        files_metadata.append(
            FileMetadata(
                file_id=file_id,
                filename=file.filename,
                content_type=file.content_type,
                thumbnail_b64=base64.b64encode(file_data).decode("utf-8"),
            )
        )

    context = {"files": files_metadata}
    return HTMXTemplate(template_name="reorder_form.html", context=context)


@post("/submit")
async def submit_ordered_files(request: Request) -> Template:
    """
    Accepts ordered file IDs, fetches the files from Redis, and enqueues one job per file.
    """
    form = await request.form()
    ordered_files: list[str] = form.getall("file")

    for idx, file_id in enumerate(ordered_files):
        is_first_page = idx == 0
        q.enqueue(
            run_inference,
            get_file_key(file_id),
            is_first_page,
            job_id=file_id,
            result_ttl=REDIS_TIMEOUT,
            ttl=REDIS_TIMEOUT,
            failure_ttl=REDIS_TIMEOUT,
        )

    context = {"status": "waiting for status...", "order": ",".join(ordered_files)}
    return HTMXTemplate(template_name="fragments/status.html", context=context)
