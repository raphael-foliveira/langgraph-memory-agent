from langchain_community.document_loaders import RecursiveUrlLoader
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def preprocess_documents(
    url: str,
    max_depth: int = 2,
) -> list[Document]:
    if os.path.exists("doc_splits.json"):
        with open("doc_splits.json", "r") as f:
            print("Loading doc_splits.json")
            doc_splits = json.load(f)
            if len(doc_splits) > 0:
                print("Loaded doc_splits.json")
                return [Document.model_validate(split) for split in doc_splits]
    print("Loading docs from url")
    docs = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
    ).load()

    split_documents = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200,
        chunk_overlap=50,
    ).split_documents(docs)
    with open("doc_splits.json", "w") as f:
        print("Saving doc_splits.json")
        json.dump([doc.model_dump() for doc in split_documents], f)
    print("Saved doc_splits.json")
    return split_documents
