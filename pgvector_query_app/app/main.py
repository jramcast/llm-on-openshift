import os
from fastapi import FastAPI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

app = FastAPI()

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not CONNECTION_STRING or not COLLECTION_NAME:
    raise RuntimeError("Specify CONNECTION_STRING and COLLECTION NAME env variables")


@app.get("/find")
def find(q: str):
    store = connect_to_db()
    results = store.similarity_search(q, k=3, return_metadata=True)
    return results


def connect_to_db():
    embeddings = HuggingFaceEmbeddings()
    return PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings)

