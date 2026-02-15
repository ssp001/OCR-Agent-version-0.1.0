"""
Docstring for app.scripts.aibase
aibase run agent direct for api endpoint only.
"""
from core.src.mcp.src.mcpagent import McpAgent
from langsmith import traceable
import logging


@traceable(name="AiBase")
class AiBase:
    """
    Docstring for AiBase
    ## Args
    - verbose: verbose is an agen thinking process you can see what the agent is thinking on ouput genaration if verbose is true.
    - memory_enable: memory_enable is like agent memory system if it is set as true aent can know past conversation.
    - retry_on_error: agent will retry on mcp delay or any exception occured on running. 
    """

    def __init__(self, verbose: bool | None, memory_enable: bool | None, retry_on_error: bool | None, model: str):
        self.verbose = verbose
        self.memory_enable = memory_enable
        self.retry_on_error = retry_on_error
        self.model = model
        self.Agent_invoke = McpAgent(llm_model=self.model,
                                     memory_enable=self.memory_enable, verbose=self.verbose, retry_on_error=self.retry_on_error)
        if not isinstance(retry_on_error, bool):
            raise TypeError("retry_on_error should be bolean type")
        if not isinstance(memory_enable, bool):
            raise TypeError("memory_enable should be boolean type")
        if not isinstance(model, str):
            raise TypeError("model should be string type")
        if not isinstance(verbose, bool):
            raise TypeError("verbose should be boolean type")

    """
    Docstring for ai_run
    ## Args
    - query: type you quary here use it in direct in api endpoint
    """

    async def as_run(self, query: str):
        if query is not str:
            return TypeError("value should be string value")
        try:
            result = await self.Agent_invoke.run_as_agent(query)
            logging.info("Your agent qury is excuted")
            return result
        except Exception as e:
            logging.exception("The ai base facing a problem")
            raise RuntimeError(f"{e}")
