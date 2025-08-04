from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langmem.short_term import SummarizationNode
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage


def create_agent_graph(
    response_model: Runnable,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    tools: list[Tool] = [],
) -> CompiledStateGraph:
    def call_tool_or_respond(state: MessagesState):
        response = response_model.invoke(input=state["messages"])
        return {"messages": [AIMessage.model_validate(response.model_dump())]}

    workflow = StateGraph(MessagesState)

    workflow.add_node(node=call_tool_or_respond)
    workflow.add_node(
        node="summarization",
        action=SummarizationNode(
            model=response_model,
            max_tokens=1024,
            max_tokens_before_summary=8000,
            output_messages_key="messages",
        ),
    )

    workflow.add_edge(start_key=START, end_key="summarization")
    workflow.add_edge(start_key="summarization", end_key=call_tool_or_respond.__name__)
    if len(tools) > 0:
        workflow.add_node(node="tools", action=ToolNode(tools))
        workflow.add_conditional_edges(
            source=call_tool_or_respond.__name__,
            path=tools_condition,
            path_map={
                "tools": "tools",
                END: END,
            },
        )
        workflow.add_edge(start_key="tools", end_key=call_tool_or_respond.__name__)
    else:
        workflow.add_edge(start_key=call_tool_or_respond.__name__, end_key=END)

    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
    )
