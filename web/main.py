from typing import Union
from fastapi import FastAPI

# from worker.inference.inference import run_inference

app = FastAPI()


@app.get("/")
def index():
    # run_inference(["worker/test_imgs/img1.JPG", "worker/test_imgs/img2.JPG"])
    return {"Hello": "world"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
