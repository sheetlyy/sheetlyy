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
from rq.job import Job
from worker.utils.image_preprocessing import preprocess_image_from_bytes
from worker.main import run_inference


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


@get("/add-file", sync_to_thread=False)
def add_file_input(index: int) -> Template:
    is_max_reached = index + 1 >= 10
    context = {"index": index + 1, "is_max_reached": is_max_reached}
    return HTMXTemplate(template_name="fragments/file_input.html", context=context)


# TODO:
# - enlarge/rotate/remove/add images
# - force download button
# - back to home button
# - 100MB limit info text
# - more messages while processing (estim completion time?)


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
        # encode image bytes as string using base64,
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
    r.set(f"upload:{upload_id}", json.dumps(files_redis), ex=60 * 30)

    context = {"upload_id": upload_id, "files": list(files.items())}
    return HTMXTemplate(template_name="reorder_form.html", context=context)


@post("/submit/{upload_id:str}")
async def submit_ordered_files(request: Request, upload_id: str) -> Template:
    """
    Accepts ordered file IDs, fetches the files from Redis, and enqueues the job.

    TODO: fix this shit
    """
    files_redis = r.get(f"upload:{upload_id}")
    if not files_redis:
        raise HTTPException(
            "Uploaded files have expired, please reupload and try again",
            status_code=404,
        )

    files: dict = json.loads(files_redis.decode("utf-8"))  # type: ignore
    for file_id, file_data in files.items():
        files[file_id] = base64.b64decode(file_data)

    form = await request.form()
    ordered_files = form.getall("file")

    image_bytes = [files.get(file_id) for file_id in ordered_files]

    job = q.enqueue(run_inference, image_bytes)

    r.delete(f"upload:{upload_id}")

    context = {"job_id": job.id}
    return HTMXTemplate(template_name="fragments/uploaded.html", context=context)


@get("/status/{job_id:str}", sync_to_thread=False)
def get_status(job_id: str) -> Template:
    job = Job.fetch(id=job_id, connection=r)

    is_failed = any([job.is_failed, job.is_stopped, job.is_canceled])
    if job.is_finished:
        status = "Job complete, downloading..."
    elif is_failed:
        status = "Failed to complete job."
    else:
        status = "Processing..."

    is_complete = job.is_finished or is_failed
    context = {"status": status, "is_finished": job.is_finished, "job_id": job_id}
    return HTMXTemplate(
        template_name="fragments/status.html",
        context=context,
        status_code=286 if is_complete else 200,  # 286 stops htmx polling
    )


@get("/download/{job_id:str}", sync_to_thread=False)
def download_file(job_id: str) -> Response:
    job = Job.fetch(id=job_id, connection=r)

    result = job.return_value()
    if result is None:
        raise HTTPException(detail="File not found", status_code=404)

    file_data = result["file_data"]
    filename = result["filename"]

    return Response(
        content=file_data,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
