import argparse
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent


CHUNK_SIZE = 256

#loading the codebase as documents
def load_codebase(repo_path: str) -> list:
    docs = []
    for path in Path(repo_path).rglob("*.py"):
        text = path.read_text(encoding="utf-8" , errors="ignore")
        docs.append(Document(page_content=text , metadata={"source": str(path)}))
    return docs 

#chunking the loaded codebase 
def chunk_code(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size = CHUNK_SIZE ,
        chunk_overlap = 32
    )
    return splitter.split_documents(docs)

#embedding the chunks and storing in db
def build_vector_store(chunks: list) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name ="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks,embedding=embeddings)

def build_agent(vector_store: Chroma):
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={'k': 4}),
        name = "search_codebase",
        description="search the codebase for relevant function, classes , logic",
    )
    #initilize the llm
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )
    return create_agent(
        llm , tools = [retriever_tool],
        system_prompt=(
            "you are a senior engineer . always use search_codebase before answering"
            "reference specific file and function name"
            "if not found say 'nhi milla bey' ",
        )
    )




