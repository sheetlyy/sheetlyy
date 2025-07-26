from litestar import Litestar, MediaType, get
# from worker.inference.inference import run_inference


@get("/", sync_to_thread=False)
def index() -> str:
    # run_inference(["worker/test_imgs/img1.JPG", "worker/test_imgs/img2.JPG"])
    return "Hello, world!"


@get("/books/{book_id:int}", sync_to_thread=False)
def get_book(book_id: int) -> dict[str, int]:
    return {"book_id": book_id}


@get(path="/health-check", media_type=MediaType.TEXT, sync_to_thread=False)
def health_check() -> str:
    return "healthy"


app = Litestar([index, get_book, health_check])
