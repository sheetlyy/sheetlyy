import uuid

from litestar import get
from litestar.response import Template, Response
from litestar.plugins.htmx import HTMXTemplate
from litestar.exceptions import HTTPException

from rq.job import Job
from web.utils.clients import r
from web.utils.constants import REDIS_TIMEOUT
from worker.main import generate_musicxml


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
    r.set(f"download:{download_id}", xml_str, ex=REDIS_TIMEOUT)

    context = {"download_id": download_id}
    return HTMXTemplate(template_name="fragments/download.html", context=context)


@get("/download/{download_id:str}", sync_to_thread=False)
def download_file(download_id: str) -> Response:
    file_data = r.get(f"download:{download_id}")
    if not file_data:
        raise HTTPException(
            "Download not found, please reupload and try again",
            status_code=404,
        )

    filename = "result.musicxml"

    return Response(
        content=file_data,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
