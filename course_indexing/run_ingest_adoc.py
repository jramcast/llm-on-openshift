import os
import re
from pathlib import Path
from scaffolding.outline import load_outline, Outline
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)

PROJECTS_DIR = os.getenv("PROJECTS_DIR", ".projects")

COLLECTION_NAME = "courses_adoc"
CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordb@postgresql:5432/vectordb"
skus = os.listdir(PROJECTS_DIR)
# headers_to_split_on = [("h3", "Header 1")]
# markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

headers_to_split_on = [
    ("h1", "course"),
    ("h2", "chapter"),
    ("h3", "section"),
    # ("h4", "subsection"),
]
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


# Chapter title example: "Chapter 1. Kustomize Overlays"
chapter_title_re = r"Chapter \d+\. (.+)"
# Section title example: "1.3. Kustomize Overlays"
section_title_re = r"\d+\.\d+\. (.+)"


def find_section_id_from_split_metadata(metadata: dict, outline: Outline):
    chapter = metadata.get("chapter", "")
    section = metadata.get("section", "")

    chapter_match = re.search(chapter_title_re, chapter)
    chapter_title = chapter_match.group(1) if chapter_match else ""

    section_match = re.search(section_title_re, section)
    section_title = section_match.group(1) if section_match else ""

    return outline.find_section_id_by_title(chapter_title, section_title)


def load_splits(sku):
    project_path = Path(PROJECTS_DIR) / sku
    # loader = DirectoryLoader(
    #     project_path / "content", glob="**/*.adoc", loader_cls=TextLoader
    # )
    loader = TextLoader(
        "/home/jairamir/dev/llm-on-openshift/course_indexing/.projects/DO280/DO280.en-US.html"
    )
    docs = loader.load()
    outline = load_outline(project_path / "outline.yml")

    print(sku)
    for c in outline.chapters:
        print(c.title, c.number)

        for s in c.sections:
            print("\t", s)

    md_header_splits = []
    for doc in docs:
        print(doc.metadata)
        # print(doc.page_content[:20])
        splits = html_splitter.split_text(doc.page_content)
        for s in splits:
            add_metadata(
                s,
                {
                    "sku": sku,
                    "section_id": find_section_id_from_split_metadata(
                        s.metadata, outline
                    ),
                },
            )
            print("-- Metadata", s.metadata)
            print(s.page_content[:1000])
            print("**" * 40)
        exit()
        splits = [clean_split(s) for s in splits]
        splits = [
            add_metadata(s, {"sku": sku, "file": doc.metadata["source"]})
            for s in splits
        ]
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
    split.page_content = split.page_content.replace("\x00", "").replace(
        ":gls_prefix", ""
    )
    return split


def add_metadata(split, metadata):
    split.metadata.update(metadata)
    return split


def is_valid_spit(split):
    return (
        "~]$ *lab start" not in split.page_content
        and "~]$ *lab finish" not in split.page_content
        and len(split.page_content) > 10
    )


# embeddings = HuggingFaceEmbeddings()

# store = PGVector(
#     connection_string=CONNECTION_STRING,
#     collection_name=COLLECTION_NAME,
#     embedding_function=embeddings,
# )


# def ingest_course(sku):
#     print(f"Loading {sku}")
#     store.add_documents(load_splits(sku))
#     print("Splits added to PGvector store")


if __name__ == "__main__":
    for sku in skus:
        load_splits(sku)
        # print(f"Loading {sku}")
        # store.add_documents(load_splits(sku))
        # print("Splits added to PGvector store")
