from langgraph.store.base import BaseStore


class LongTermMemory:
    def __init__(self, store: BaseStore, thread_id: str):
        self.store = store
        self.thread_id = thread_id

    async def save_memory(self, key: str, data: str) -> str:
        """Saves a memory to the long term memory database"""
        await self.store.aput(
            namespace=("user", self.thread_id),
            key=key,
            value={"data": data},
        )
        return "information saved successfully"

    async def erase_memory(self, key: str) -> str:
        """Erases a memory from the long term memory database"""
        await self.store.adelete(
            namespace=("user", self.thread_id),
            key=key,
        )
        return "information deleted successfully"

    async def retrieve_memory(self, query: str) -> str:
        """Retrieves a memory from the long term memory database"""
        memories = await self.store.asearch(("user", self.thread_id), query=query)
        return "\n".join([f"{m.key}: {m.value['data']}" for m in memories])
