import asyncio
from app.core.src.router.src.router import RouterAi
from app.graph.src.aigraph import ocr
from typing import *

ai = RouterAi(model="mistralai/Mistral-7B-Instruct-v0.2",
              verbose=True, temprathure=0.0)


async def run():
    result = await ocr.ainvoke({"user_query": "Pull an up-to-date hourly forecast of nabadwip(next 12 hours).",
                                "router_output": {}, "final_answer": None})
    print("AI RESULT ↓")
    print(result["final_answer"])


asyncio.run(run())
