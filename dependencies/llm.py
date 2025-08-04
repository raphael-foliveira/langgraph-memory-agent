from langgraph.store.base import BaseStore
from langgraph.store.postgres.base import PostgresIndexConfig
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from langgraph.checkpoint.base import BaseCheckpointSaver
from llm import create_agent_graph
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from langchain_postgres import PGVector
from fastapi import Depends, Path
from llm.memory import LongTermMemory
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from pydantic import SecretStr

_environment: dict[str, str] = {}


async def get_postgres_dsn():
    return _cached_get_env("DATABASE_URL")


async def get_vector_db_dsn():
    return _cached_get_env("VECTOR_DB_URL")


async def get_async_db_engine(dsn: str = Depends(get_vector_db_dsn)):
    return create_async_engine(dsn)


async def get_db_engine(dsn: str = Depends(get_vector_db_dsn)):
    return create_engine(dsn)


async def get_google_api_key():
    return SecretStr(_cached_get_env("GEMINI_API_KEY"))


async def get_google_embedding_model():
    return _cached_get_env(
        "GOOGLE_EMBEDDING_MODEL",
        default="gemini-embedding-001",
    )


async def get_google_generative_ai_model():
    return _cached_get_env(
        "GOOGLE_GENERATIVE_AI_MODEL",
        default="models/gemini-2.5-flash",
    )


async def get_google_generative_ai_embeddings(
    model: str = Depends(get_google_embedding_model),
    google_api_key: SecretStr = Depends(get_google_api_key),
):
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=google_api_key,
    )


async def get_postgres_vector_store(
    async_engine: AsyncEngine = Depends(get_async_db_engine),
    embeddings=Depends(get_google_generative_ai_embeddings),
):
    return PGVector(
        connection=async_engine,
        embeddings=embeddings,
        collection_name="tibia_data",
    )

async def get_postgres_store_config(
    embeddings: GoogleGenerativeAIEmbeddings = Depends(
        get_google_generative_ai_embeddings
    ),
) -> PostgresIndexConfig:
    return {
        "ann_index_config": {"kind": "ivfflat"},
        "embed": embeddings,
        "dims": 3072,
    }

async def get_postgres_store(
    dsn: str = Depends(get_postgres_dsn),
    config: PostgresIndexConfig = Depends(get_postgres_store_config),
):
    async with AsyncPostgresStore.from_conn_string(
        dsn,
        index=config,
    ) as store:
        yield store


async def get_postgres_checkpointer(dsn: str = Depends(get_postgres_dsn)):
    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        yield checkpointer


async def get_long_term_memory(
    store: BaseStore = Depends(get_postgres_store),
    thread_id: str = Path(...),
):
    return LongTermMemory(store, thread_id)


async def get_google_genai_chat(
    model: str = Depends(get_google_generative_ai_model),
    google_api_key: SecretStr = Depends(get_google_api_key),
):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=google_api_key,
    )


async def get_rag_retriever_tool(
    vector_store: PGVector = Depends(get_postgres_vector_store),
):
    return create_retriever_tool(
        retriever=vector_store.as_retriever(),
        name="retrieve_relevant_information",
        description="Retrieve relevant information from the vector store",
    )


async def get_tools(
    long_term_memory: LongTermMemory = Depends(get_long_term_memory),
):
    return [
        long_term_memory.erase_memory,
        long_term_memory.save_memory,
        long_term_memory.retrieve_memory,
    ]


async def get_google_genai_chat_with_tools(
    google_genai_chat: ChatGoogleGenerativeAI = Depends(get_google_genai_chat),
    tools: list[Tool] = Depends(get_tools),
):
    return google_genai_chat.bind_tools(tools)


async def get_main_graph(
    google_genai_with_tools: Runnable = Depends(get_google_genai_chat_with_tools),
    checkpointer: BaseCheckpointSaver = Depends(get_postgres_checkpointer),
    store: BaseStore = Depends(get_postgres_store),
    tools: list[Tool] = Depends(get_tools),
):
    return create_agent_graph(
        response_model=google_genai_with_tools,
        checkpointer=checkpointer,
        store=store,
        tools=tools,
    )


def _cached_get_env(
    key: str,
    *,
    optional: bool = False,
    default: str | None = None,
) -> str:
    if key in _environment:
        return _environment[key]
    value = os.getenv(key)
    if value is None and not optional and default is None:
        raise ValueError(f"{key} is not set")
    _environment[key] = value or default or ""
    return _environment[key]
