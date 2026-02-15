"""
Docstring for app.core.src.router.router
## Router ai
- this is a router ai that will inhance the ai output 
"""
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langsmith import traceable
from typing import List
import logging
import os
load_dotenv()


@traceable(name="router-ai")
class RouterAi:
    """
    Docstring for RouterAi

    :var task: a Router-ai that will help the main ai to inhance the output.
    :var Rules: None
    :var schema: None
    :var query: None
    """

    def __init__(self, model: str, verbose: bool, temprathure: float):
        try:
            self.huggingface_instance = HuggingFaceEndpoint(
                model=model, verbose=verbose, huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), temperature=temprathure)
            self.llm_endpoint = ChatHuggingFace(llm=self.huggingface_instance)

            self.parser = StrOutputParser()
            self.prompt = self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
            You are an intent analysis and planning AI.

            Your task:
            - Understand the user's intent
            - Understand the user's emotion
            - Understand the user's Query if that qury needs Rag operation. 
            - Extract the key points required to answer the query
            - Decide whether external knowledge retrieval is required

            Rules:
            - Return ONLY valid JSON
            - Do NOT include explanations
            - Do NOT include markdown
            - Do NOT answer the user
            - Do NOT include extra text

            Return JSON in EXACTLY this schema:

            {{
            "intent": "string",
            "emotion":"string"
            "needs_Rag-Operation":true | flase
            "needs_context": true | false,
            "key_points": ["string"],
            "search_queries": ["string"],
            "answer_style": "string"
            }}
            """
                    ),
                    ("human", "{input}")
                ]
            )

            self.router_runner = self.prompt | self.llm_endpoint | self.parser
        except Exception as error:
            logging.exception("RouterAI execution failed ")
            raise RuntimeError(error)

    async def run_session(self, query_input: str) -> List[str]:
        """
        Docstring for run_session

        :param self: None
        :param input: there is the varriable of your string user query
        :type input: str
        :return: this class will return a list of string that will inhance the output of the ai. 
        :rtype: List[str]
        """
        try:
            raw_output = await self.router_runner.ainvoke({"input": query_input})
            logging.info("the agent output is fecthed")
            return raw_output
        except Exception as e:
            logging.exception("there is a error in ai_router")
            raise RuntimeError(e)
