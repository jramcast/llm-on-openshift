import os
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

load_dotenv()

app = FastAPI()

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not CONNECTION_STRING or not COLLECTION_NAME:
    raise RuntimeError("Specify CONNECTION_STRING and COLLECTION NAME env variables")


embeddings = HuggingFaceEmbeddings()


@app.get("/find")
def find(q: str, min_score=0.5):
    store = connect_to_db()
    results = store.similarity_search_with_relevance_scores(q, k=3, score_threshold=float(min_score))
    return results


def connect_to_db():
    return PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings)
