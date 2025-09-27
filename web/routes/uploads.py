import uuid
import json
import base64
from dataclasses import dataclass
from typing import Annotated

from litestar import post, Request
from litestar.response import Template
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.plugins.htmx import HTMXTemplate
from litestar.exceptions import HTTPException

from redis import Redis
from web.utils.clients import r, q
from web.utils.constants import REDIS_TIMEOUT
from worker.utils.image_preprocessing import preprocess_image_from_bytes
from worker.main import run_inference


@dataclass
class FileMetadata:
    """
    File metadata used for rendering file thumbnails in the drag-and-drop UI.
    """

    file_id: str
    filename: str
    content_type: str
    thumbnail_b64: str


def compress_and_encode_image(img_bytes: bytes) -> str:
    """
    Compresses and encodes image bytes to a base64 string that is JSON-serializable
    for storage in Redis.
    """
    compressed_bytes = preprocess_image_from_bytes(img_bytes)
    compressed_b64 = base64.b64encode(compressed_bytes).decode("utf-8")
    return compressed_b64


def save_uploads(uploads: dict[str, str], r: Redis) -> str:
    """
    JSON-serializes uploaded files to Redis under a unique upload ID.
    """
    upload_id = str(uuid.uuid4())
    r.set(f"upload_files:{upload_id}", json.dumps(uploads), ex=REDIS_TIMEOUT)
    return upload_id


def load_uploads(upload_id: str, r: Redis) -> dict[str, bytes]:
    """
    Loads, deserializes, and decodes uploaded files from Redis under the given upload ID.
    """
    serialized_files = r.get(f"upload_files:{upload_id}")
    if not serialized_files:
        raise HTTPException(
            "Uploaded files have expired, please reupload and try again",
            status_code=404,
        )
    r.delete(f"upload_files:{upload_id}")

    # need to decode manually here since we do not use decode_responses=True
    serialized_files_str = serialized_files.decode("utf-8")  # type: ignore

    files_dict: dict[str, str] = json.loads(serialized_files_str)
    files: dict[str, bytes] = {
        file_id: base64.b64decode(file_str) for file_id, file_str in files_dict.items()
    }

    return files


@post("/upload")
async def handle_file_uploads(
    data: Annotated[list[UploadFile], Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Template:
    """
    Accepts files from user, compresses them, stores them temporarily in Redis,
    and returns a view that allows the user to reorder files.
    """
    compressed_files: dict[str, str] = {}
    files_metadata: list[FileMetadata] = []
    for file in data:
        file_id = str(uuid.uuid4())
        file_data = await file.read()

        compressed_files[file_id] = compress_and_encode_image(file_data)
        files_metadata.append(
            FileMetadata(
                file_id=file_id,
                filename=file.filename,
                content_type=file.content_type,
                thumbnail_b64=base64.b64encode(file_data).decode("utf-8"),
            )
        )

    upload_id = save_uploads(compressed_files, r)

    context = {"upload_id": upload_id, "files": files_metadata}
    return HTMXTemplate(template_name="reorder_form.html", context=context)


@post("/submit/{upload_id:str}")
async def submit_ordered_files(request: Request, upload_id: str) -> Template:
    """
    Accepts ordered file IDs, fetches the files from Redis, and enqueues one job per file.
    """
    files = load_uploads(upload_id, r)

    form = await request.form()
    ordered_files: list[str] = form.getall("file")

    for idx, file_id in enumerate(ordered_files):
        file_bytes = files.get(file_id)
        if file_bytes is None:
            raise HTTPException("Invalid file ID, file not found", status_code=404)
        is_first_page = idx == 0
        q.enqueue(
            run_inference,
            file_bytes,
            is_first_page,
            job_id=file_id,
            result_ttl=REDIS_TIMEOUT,
            ttl=REDIS_TIMEOUT,
            failure_ttl=REDIS_TIMEOUT,
        )

    context = {"status": "waiting for status...", "order": " ".join(ordered_files)}
    return HTMXTemplate(template_name="fragments/status.html", context=context)
