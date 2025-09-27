import os
from redis import Redis
from rq import Queue
from web.utils.constants import REDIS_TIMEOUT


r = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
q = Queue(connection=r, default_timeout=REDIS_TIMEOUT)
