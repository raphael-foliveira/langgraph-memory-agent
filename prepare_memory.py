import asyncio
from dependencies.llm import (
    get_postgres_dsn,
    get_google_generative_ai_embeddings,
    get_google_embedding_model,
    get_google_api_key,
    get_postgres_store_config,
)
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


async def main():
    postgres_dsn = await get_postgres_dsn()
    embeddings = await get_google_generative_ai_embeddings(
        model=await get_google_embedding_model(),
        google_api_key=await get_google_api_key(),
    )
    vector_store_config = await get_postgres_store_config(embeddings=embeddings)

    async with (
        AsyncPostgresStore.from_conn_string(
            postgres_dsn, index=vector_store_config
        ) as store,
        AsyncPostgresSaver.from_conn_string(postgres_dsn) as checkpointer,
    ):
        await store.setup()
        await checkpointer.setup()


if __name__ == "__main__":
    asyncio.run(main())
