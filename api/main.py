from typing import Dict, Any
from fastapi import FastAPI, Request
from model import DataEntry

app = FastAPI()


@app.post("/")
async def root(entry: DataEntry):
    # data: Dict[str, Any] = await request.json()
    print(entry)

    return data

