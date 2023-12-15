import os
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread

import gradio as gr
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.vectorstores.pgvector import PGVector

load_dotenv()

# Parameters

APP_TITLE = os.getenv('APP_TITLE', 'Talk with Red Hat Training')

INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TOP_K = int(os.getenv('TOP_K', 10))
TOP_P = float(os.getenv('TOP_P', 0.95))
TYPICAL_P = float(os.getenv('TYPICAL_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', 1.03))

DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_COLLECTION_NAME = os.getenv('DB_COLLECTION_NAME')

# Streaming implementation
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        metadata_as_str = metadata_to_string(item.metadata)
        if metadata_as_str not in unique_list:
            unique_list.append(metadata_as_str)
    return unique_list

def metadata_to_string(metadata):
    source = metadata.get("source", "")
    source = source.replace("pdf/", "")
    # Start counting pages at 1, not 0. If no page is given, then display 0
    page = metadata.get("page", -1) + 1
    return f"{source}, page {page}"

def stream(input_text) -> Generator:
    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        resp = qa_chain({"question": input_text})
        sources = remove_source_duplicates(resp.get('source_documents', []))
        if len(sources) != 0:
            q.put("\n*Sources:* \n")
            for source in sources:
                q.put("* " + str(source) + "\n")
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

# A Queue is needed for Streaming implementation
q = Queue()

############################
# LLM chain implementation #
############################

# Document store: pgvector vector store
embeddings = HuggingFaceEmbeddings()
store = PGVector(
    connection_string=DB_CONNECTION_STRING,
    collection_name=DB_COLLECTION_NAME,
    embedding_function=embeddings)

# LLM
llm = HuggingFaceTextGenInference(
    inference_server_url=INFERENCE_SERVER_URL,
    max_new_tokens=MAX_NEW_TOKENS,
    top_k=TOP_K,
    top_p=TOP_P,
    typical_p=TYPICAL_P,
    temperature=TEMPERATURE,
    repetition_penalty=REPETITION_PENALTY,
    streaming=True,
    verbose=False,
    callbacks=[QueueCallback(q)]
)

# Memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

def get_chat_history(history):
    messages = []
    for msj in history:

        if msj.type == "human":
            messages.append(f"<s>[INST] {msj.content} [/INST]")
        else:
            messages.append(f"{msj.content}</s>")

    return "\n".join(messages)

# Prompt
template="""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named RedHatTrainingBot answering questions about the Red Hat products and technologies ecosystem.
You will be given a question you need to answer, a context to provide you with information, and the chat history. You must answer the question based as much as possible on this context.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

This is your CONTEXT for answer the last question:
{context}
<</SYS>>

Who are you?
[/INST]
I am RedHatTrainingBot, a helpful, respectful and honest assistant that can ask technical questions about the Red Hat products and technologies ecosystem.
</s>
{chat_history}
<s>
[INST]
{question}
[/INST]
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

condense_template = """\
<s>[INST] You must echo the user input. Repeat the user input and do not change a single character of the message.
Do not add any additional characters. Just echo the user message as is.
For example, if write "Hello, good Morning?", then you must anwser "Hello, good Morning?".
[/INST]
Understood, I will echo every single message.
</s>
<s>
[INST]
how are you?
[/INST]
how are you?
</s>
<s>
[INST]
what is a service?
[/INST]
what is a service?
</s>
<s>
[INST]
{question}
[/INST]
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

condense_question_llm = HuggingFaceTextGenInference(
    inference_server_url=INFERENCE_SERVER_URL,
    max_new_tokens=MAX_NEW_TOKENS,
    top_k=TOP_K,
    top_p=TOP_P,
    typical_p=TYPICAL_P,
    temperature=TEMPERATURE,
    repetition_penalty=REPETITION_PENALTY,
    streaming=False,
    verbose=False,
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.2 }
    ),
    memory=memory,
    verbose=True,
    get_chat_history=get_chat_history,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    condense_question_llm=condense_question_llm,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    return_source_documents=True
)

# Gradio implementation
def ask_llm(message, history):
    for next_token, content in stream(message):
        yield(content)

with gr.Blocks(title="RedHatTrainingBot", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(None,'assets/robot-head.svg'),
        render=False,
        show_copy_button=True,
    )
    gr.ChatInterface(
        ask_llm,
        chatbot=chatbot,
        #clear_btn=True,
        retry_btn=None,
        undo_btn=None,
        stop_btn=None,
        description=APP_TITLE
    )

if __name__ == "__main__":

    auth_user = os.getenv("AUTH_USER")
    auth_password = os.getenv("AUTH_PASSWORD")
    if auth_user and auth_password:
        print("Enabling basic authentication with AUTH_USER/AUTH_PASSWORD")
        auth = (auth_user, auth_password)
    else:
        auth = None

    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico',
        auth=auth
    )
