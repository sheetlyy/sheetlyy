from litestar import get
from litestar.response import Template
from litestar.plugins.htmx import HTMXTemplate

from rq.job import Job, JobStatus
from web.utils.clients import r


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
