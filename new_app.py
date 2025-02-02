from langchain import PromptTemplate
from langchain.chains import retrieval_qa
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import timeit
import sys

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

import pinecone

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-east-1-aws",
)

index = pinecone.Index('langchainpinecones')


def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents

extracted_data = load_pdf_file(data='DATA/')


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


text_chunks = text_split(extracted_data)
print("Length of my chunk:", len(text_chunks))

def download_hugging_face_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return embeddings
