import os
import uuid
import json
import base64
from typing import Annotated

from litestar import get, post, Request
from litestar.response import Template, Response
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.plugins.htmx import HTMXTemplate
from litestar.exceptions import HTTPException

from redis import Redis
from rq import Queue
from rq.job import Job, JobStatus
from worker.utils.image_preprocessing import preprocess_image_from_bytes
from worker.main import run_inference, generate_musicxml


r = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=os.getenv("REDIS_SSL", "false") == "true",
)
q = Queue(connection=r, default_timeout=60 * 30)


@get("/", sync_to_thread=False)
def index() -> Template:
    context = {"index": "1"}
    return HTMXTemplate(template_name="base.html", context=context)


# TODO: refactor, styling


@post("/upload")
async def handle_file_uploads(
    data: Annotated[list[UploadFile], Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Template:
    """
    Accepts files from user, compresses them, stores them temporarily in Redis,
    and returns a view that allows the user to reorder files.

    TODO: redo this shit
    """
    upload_id = str(uuid.uuid4())
    files: dict[str, dict[str, str | bytes]] = {}
    for file in data:
        file_id = str(uuid.uuid4())
        file_data = await file.read()
        # encode image bytes to base64 string,
        # serialize that to json, and save that to redis
        img_bytes = preprocess_image_from_bytes(file_data)
        img_str = base64.b64encode(img_bytes).decode("utf-8")
        img_b64 = base64.b64encode(file_data).decode("utf-8")

        files[file_id] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "img_bytes": img_str,
            "img_b64": img_b64,
        }

    files_redis = {file_id: info["img_bytes"] for file_id, info in files.items()}
    r.set(f"upload_files:{upload_id}", json.dumps(files_redis), ex=60 * 30)

    context = {"upload_id": upload_id, "files": list(files.items())}
    return HTMXTemplate(template_name="reorder_form.html", context=context)


@post("/submit/{upload_id:str}")
async def submit_ordered_files(request: Request, upload_id: str) -> Template:
    """
    Accepts ordered file IDs, fetches the files from Redis, and enqueues one job per file.

    TODO: fix this shit
    """
    files_redis = r.get(f"upload_files:{upload_id}")
    if not files_redis:
        raise HTTPException(
            "Uploaded files have expired, please reupload and try again",
            status_code=404,
        )
    r.delete(f"upload_files:{upload_id}")

    files_dict: dict[str, str] = json.loads(files_redis.decode("utf-8"))  # type: ignore
    files: dict[str, bytes] = {}
    for file_id, file_str in files_dict.items():
        files[file_id] = base64.b64decode(file_str)

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
            result_ttl=60 * 30,
            ttl=60 * 30,
            failure_ttl=60 * 30,
        )

    context = {"status": "Waiting for status...", "order": " ".join(ordered_files)}
    return HTMXTemplate(template_name="fragments/status.html", context=context)


@get("/status", sync_to_thread=False)
def get_status(order: str) -> Template:
    ordered_files: list[str] = order.split(" ")

    job_statuses: list[JobStatus] = []
    for file_id in ordered_files:
        job = Job.fetch(id=file_id, connection=r)
        job_statuses.append(job.get_status())

    num_finished = sum(status == JobStatus.FINISHED for status in job_statuses)
    all_finished = num_finished == len(job_statuses)
    any_failed = any(
        status in [JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELED]
        for status in job_statuses
    )

    if all_finished:
        status = "Complete, preparing for download..."
    elif any_failed:
        status = "Failed to complete. Please reupload and try again."
    else:
        status = (
            f"Processing... ({num_finished} of {len(job_statuses)} page(s) completed)"
        )

    if all_finished:
        context = {"order": order}
        return HTMXTemplate(template_name="fragments/preparing.html", context=context)
    else:
        context = {"status": status, "order": order}
        return HTMXTemplate(template_name="fragments/status.html", context=context)


@get("/prepare", sync_to_thread=False)
def prepare_download(order: str) -> Template:
    ordered_files: list[str] = order.split(" ")

    pages = []
    for file_id in ordered_files:
        page_job = Job.fetch(id=file_id, connection=r)
        result = page_job.return_value()
        if result is None:
            raise HTTPException(detail="File not found", status_code=404)
        page = result["page"]
        pages.append(page)

    xml_bytes = generate_musicxml(pages)
    xml_str = xml_bytes.decode("utf-8")
    download_id = str(uuid.uuid4())
    r.set(f"download:{download_id}", xml_str, ex=60 * 30)

    context = {"download_id": download_id}
    return HTMXTemplate(template_name="fragments/download.html", context=context)


@get("/download/{download_id:str}", sync_to_thread=False)
def download_file(download_id: str) -> Response:
    xml_redis = r.get(f"download:{download_id}")
    if not xml_redis:
        raise HTTPException(
            "Download not found, please reupload and try again",
            status_code=404,
        )
    # r.delete(f"download:{download_id}")

    # xml_str: str = xml_redis.decode("utf-8")  # type: ignore
    # xml_bytes: bytes = xml_str.encode("utf-8")
    xml_bytes = xml_redis

    file_data = xml_bytes
    filename = "result.musicxml"

    return Response(
        content=file_data,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
