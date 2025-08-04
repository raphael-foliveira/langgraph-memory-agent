from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector


def new_in_memory_vector_store(
    documents: list[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> InMemoryVectorStore:
    print("Creating in memory vector store")
    return InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
    )


def new_pgvector_store(dsn: str, embeddings: GoogleGenerativeAIEmbeddings):
    return PGVector(
        connection=dsn,
        embeddings=embeddings,
        collection_name="tibia_data",
    )


def new_retriever_tool(vector_store: VectorStore) -> Tool:
    return create_retriever_tool(
        retriever=vector_store.as_retriever(),
        name="retrieve_tibia_data",
        description="Search and return information about Tibia",
    )
