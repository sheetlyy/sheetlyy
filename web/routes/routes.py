import os
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
from worker.inference.inference import run_inference


redis_conn = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=os.getenv("REDIS_SSL", "false") == "true",
)
q = Queue(connection=redis_conn, default_timeout=60 * 30)


@get("/", sync_to_thread=False)
def index() -> Template:
    context = {"index": "1"}
    return HTMXTemplate(template_name="base.html", context=context)


@get("/add-file", sync_to_thread=False)
def add_file_input(index: int) -> Template:
    is_max_reached = index + 1 >= 10
    context = {"index": index + 1, "is_max_reached": is_max_reached}
    return HTMXTemplate(template_name="fragments/file_input.html", context=context)


# TODO: 100MB limit,


@post("/upload")
async def handle_file_uploads(
    request: Annotated[Request, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> Template:
    form = await request.form()

    ordered_files: list[UploadFile] = [
        form[file_idx] for file_idx in sorted(form, key=int)
    ]

    image_arrs = [
        preprocess_image_from_bytes(await file.read()) for file in ordered_files
    ]
    # run_inference(image_arrs)

    job = q.enqueue(run_inference, image_arrs)

    context = {"filenames": [arr.size for arr in image_arrs], "job_id": job.id}
    return HTMXTemplate(template_name="fragments/uploaded.html", context=context)


@get("/status/{job_id:str}", sync_to_thread=False)
def get_status(job_id: str) -> Template:
    job = Job.fetch(id=job_id, connection=redis_conn)

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
    job = Job.fetch(id=job_id, connection=redis_conn)

    result = job.return_value()
    if result is None:
        raise HTTPException(detail="File not found", status_code=404)

    file_bytes = result["file_bytes"]
    filename = result["filename"]

    return Response(
        content=file_bytes,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
