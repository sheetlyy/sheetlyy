from typing import Union
from fastapi import FastAPI
from worker.utils.constants import NUMBER_OF_LINES_ON_A_STAFF

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": NUMBER_OF_LINES_ON_A_STAFF}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
