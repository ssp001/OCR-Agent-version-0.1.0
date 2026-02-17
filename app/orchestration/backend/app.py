
import os
import shutil
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from typing import *
from app.orchestration.graph.src.aigraph import ocr
from app.orchestration.core.src.router.src.router import RouterAi
from fastapi import FastAPI
from pydantic import BaseModel
from app.orchestration.core.src.vector.src.vectordb import VectorDataBasePipeline
# C:\Users\shova\Desktop\project\OCR-Agent\app\orchestration\core\src\vector\src\vectordb.py


class user_input(BaseModel):
    message: str


server = FastAPI(title="Ocr-Agent",
                 summary="An agent fro daily use hard endpoint", version="0.1.0")


@server.get("/")
def Home():
    return f"hello world this is the endpoint"


ai = RouterAi(model="mistralai/Mistral-7B-Instruct-v0.2",
              verbose=True, temprathure=0.0)


@server.get("/chats")
async def stream_chat(message: user_input):

    result = await ocr.ainvoke({
        "user_query": message.message,
        "router_output": {},
        "final_answer": None
    })
    return result["final_answer"]


@server.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)  # 🔥 This fixes it

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process PDF
    VectorDataBasePipeline(file_path).connect_to_vectordb_and_store()

    return {"message": "PDF uploaded and stored successfully"}


if __name__ == "__main__":
    server.run(server, host="0.0.0.0", port=8000)
