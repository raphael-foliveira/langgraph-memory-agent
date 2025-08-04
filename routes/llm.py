from fastapi import APIRouter, Depends, Path
import json
from langchain_core.runnables import RunnableConfig
from dependencies import get_main_graph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, AnyMessage
from pydantic import BaseModel


llm_router = APIRouter()


class ChatRequest(BaseModel):
    message: str


SYSTEM_PROMPT = """
You are eager to be an expert about the user. You try to learn as much as you can about them.
"""


@llm_router.post("/chat/{thread_id}")
async def chat(
    chat_request: ChatRequest,
    graph: CompiledStateGraph = Depends(get_main_graph),
    thread_id: str = Path(...),
):
    config = RunnableConfig(configurable={"thread_id": thread_id})
    graph_state = await graph.aget_state(config=config)

    messages: list[AnyMessage] = [HumanMessage(content=chat_request.message)]

    if graph_state is None or not graph_state.values.get("messages"):
        print("initializing new thread")
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    response = await graph.ainvoke(
        input={"messages": messages},
        config=config,
    )

    with open("current_state.json", "w") as f:
        json.dump([m.model_dump() for m in response["messages"]], f)
    try:
        return AIMessage.model_validate(response["messages"][-1])

    except Exception as e:
        print("validation failed:")
        print(e)
        return response
