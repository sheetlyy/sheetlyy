from pathlib import Path

from litestar import Litestar
from litestar.static_files import create_static_files_router
from litestar.plugins.htmx import HTMXPlugin
from litestar.contrib.jinja import JinjaTemplateEngine
from litestar.template.config import TemplateConfig

from web.routes import (
    health_check,
    index,
    add_file_input,
    handle_file_uploads,
    get_status,
    download_file,
)


DEBUG = True
BASE_DIR = Path(__file__).parent

app = Litestar(
    route_handlers=[
        create_static_files_router(path="/static", directories=[BASE_DIR / "static"]),
        index,
        add_file_input,
        handle_file_uploads,
        get_status,
        download_file,
        health_check,
    ],
    debug=DEBUG,
    plugins=[HTMXPlugin()],
    template_config=TemplateConfig(
        directory=BASE_DIR / "templates", engine=JinjaTemplateEngine
    ),
)
