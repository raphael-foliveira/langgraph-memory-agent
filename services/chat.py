import json
from typing import Protocol
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph, RunnableConfig

SYSTEM_PROMPT = """
You are eager to be an expert about the user. You try to learn as much as you can about them.
"""


class MessageSaver(Protocol):
    def save(self, messages: list[AnyMessage]): ...


class FileMessageSaver:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save(self, messages: list[AnyMessage]):
        with open(self.file_path, "w") as f:
            json.dump([m.model_dump() for m in messages], f)


class ChatService:
    def __init__(
        self,
        thread_id: str,
        graph: CompiledStateGraph,
        message_saver: MessageSaver = FileMessageSaver("messages.json"),
    ) -> None:
        self.config = RunnableConfig({"configurable": {"thread_id": thread_id}})
        self.graph = graph
        self.message_saver = message_saver

    async def run(self, message: str):
        messages: list[AnyMessage] = [HumanMessage(content=message)]

        graph_state = await self.get_graph_state()
        state_messages = graph_state.values.get("messages")

        if graph_state is None or not state_messages:
            print("initializing new thread")
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

        response = await self.graph.ainvoke(
            input={"messages": messages},
            config=self.config,
        )

        self.message_saver.save(
            [
                *(state_messages if state_messages else []),
                *messages,
                response["messages"][-1],
            ]
        )

        try:
            ai_message = AIMessage.model_validate(response["messages"][-1])
            return ai_message.content
        except Exception as e:
            print("validation failed:")
            print(e)
            return response

    async def get_graph_state(self):
        return await self.graph.aget_state(self.config)
