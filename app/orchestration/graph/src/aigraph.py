# %%
from typing import *
from app.orchestration.core.src.router.src.router import RouterAi
from app.orchestration.core.src.mcp.src.mcpagent import McpAgent
from langgraph.graph import StateGraph, END
from typing import TypedDict

# -----------------------------
# 1. Define state
# -----------------------------


class AgentState(TypedDict):
    user_query: str
    router_output: Dict
    final_answer: Optional[str]

# -----------------------------
# 2. Initialize agents
# -----------------------------


router_ai = RouterAi(model="Qwen/Qwen2.5-7B-Instruct",
                     verbose=True, temprathure=0.0)
mcpagent_ai = McpAgent(llm_model="mistralai/Mistral-7B-Instruct-v0.2",
                       memory_enable=True, verbose=False, retry_on_error=True)


async def router_ai_run(state: AgentState) -> Dict:

    user_query = state.get("user_query")
    if not user_query:
        raise ValueError("user_query is missing from state")
    result = await router_ai.run_session(query_input=user_query)
    result
    return {
        "router_output": result
    }


# -----------------------------
# 3. Nodes
# -----------------------------


async def McpAgent_run(state: AgentState) -> List[str]:

    router_output = state.get("router_output")
    result = await mcpagent_ai.run_as_agent(query=router_output)
    return {"final_answer": result}


# -----------------------------
# 4. Build graph
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("router", router_ai_run)
graph.add_node("main_ai", McpAgent_run)

graph.set_entry_point("router")
graph.add_edge("router", "main_ai")
graph.add_edge("main_ai", END)

ocr = graph.compile()

# %%
