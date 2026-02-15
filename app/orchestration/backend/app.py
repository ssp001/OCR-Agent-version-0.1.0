
from app.orchestration.backend.vector import rag_pipeline
from fastapi.responses import StreamingResponse
import asyncio
from typing import *
from app.orchestration.graph.src.aigraph import ocr
from app.orchestration.core.src.router.src.router import RouterAi
from fastapi import FastAPI
from pydantic import BaseModel


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
async def stream_chat(message: str, chat_id: str):

    async def event_stream():

        prompt = rag_pipeline(message)

        async for chunk in ocr.astream({
            "user_query": prompt,
            "router_output": {},
            "final_answer": None
        }):
            if chunk.get("final_answer"):
                yield chunk["final_answer"]

    return StreamingResponse(event_stream(), media_type="text/plain")


if __name__ == "__main__":
    server.run(server, host="0.0.0.0", port=8000)
