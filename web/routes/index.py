from litestar import get
from litestar.response import Template
from litestar.plugins.htmx import HTMXTemplate

from web.utils.constants import MAX_UPLOAD_SIZE


@get("/", sync_to_thread=False)
def index() -> Template:
    context = {"max_upload_mb": MAX_UPLOAD_SIZE // 1_000_000}
    return HTMXTemplate(template_name="base.html", context=context)
