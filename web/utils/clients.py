import os
from redis import Redis
from rq import Queue
from web.utils.constants import REDIS_TIMEOUT

r = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=os.getenv("REDIS_SSL", "false") == "true",
)
q = Queue(connection=r, default_timeout=REDIS_TIMEOUT)
