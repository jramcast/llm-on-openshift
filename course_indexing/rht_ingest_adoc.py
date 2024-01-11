import os
import pathlib
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import DirectoryLoader,TextLoader

COURSES_PATH = pathlib.Path.home() / "courses"

COLLECTION_NAME = "courses_adoc"
CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordb@postgresql:5432/vectordb"
skus = os.listdir(COURSES_PATH)
headers_to_split_on = [
    ("==", "section")
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


def load_splits(sku):
    loader = DirectoryLoader(
        COURSES_PATH / sku / "content",
        glob="**/*.adoc", 
        loader_cls=TextLoader
    )
    docs = loader.load()

    md_header_splits = []
    for doc in docs:
        splits = markdown_splitter.split_text(doc.page_content)
        splits = [clean_split(s) for s in splits]
        splits = [add_metadata(s, {"sku": sku, "file": doc.metadata["source"]}) for s in splits]
        splits = [s for s in splits if is_valid_spit(s)]
        md_header_splits += splits

    chunk_size = 1024
    chunk_overlap = 128
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    all_splits = text_splitter.split_documents(md_header_splits)
    return all_splits


def clean_split(split):
    split.page_content = split.page_content.replace('\x00', "").replace(':gls_prefix', '')
    return split


def add_metadata(split, metadata):
    split.metadata.update(metadata)
    return split


def is_valid_spit(split):
    return (
        "~]$ *lab start" not in split.page_content and
        "~]$ *lab finish" not in split.page_content and
        len(split.page_content) > 10
    )


embeddings = HuggingFaceEmbeddings()

store = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings)


def ingest_course(sku):
    print(f"Loading {sku}")
    store.add_documents(load_splits(sku))
    print("Splits added to PGvector store")


if __name__ == '__main__':
    for sku in skus:
        print(f"Loading {sku}")
        store.add_documents(load_splits(sku))
        print("Splits added to PGvector store")
