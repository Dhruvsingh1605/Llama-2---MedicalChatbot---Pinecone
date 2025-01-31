# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embedding
# from langchain_pinecone import PineconeVectorStore
# from langchain_groq import ChatGroq
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# from langchain.chains import create_retrieval_chain
# import os
# from pinecone import Pinecone

# app = Flask(__name__)

# load_dotenv()

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# embeddings = download_hugging_face_embedding()

# index_name = "medicalchatbot"

# try:
#     index = pc.Index(index_name)
#     print(f"Index '{index_name}' exists.")
# except Exception as e:
#     raise ValueError(f"Error connecting to index '{index_name}': {str(e)}")

# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings,
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# llm = ChatGroq(temperature=0.4, max_tokens=500)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt.format(user_input="{input}")),  
#     ("human", "{input}"),
# ])

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# @app.route("/")
# def index():
#     return render_template("chat.html")

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     try:
#         msg = request.form["msg"]
#         response = rag_chain.invoke({"input": msg})
#         print("Response:", response["answer"])
#         return str(response["answer"])
#     except Exception as e:
#         print(f"Error processing message: {str(e)}")
#         return str("Sorry, there was an error processing your request.")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True)



import streamlit as st
from src.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
import os
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = download_hugging_face_embedding()
index_name = "medicalchatbot"

try:
    index = pc.Index(index_name)
    print(f"Index '{index_name}' exists.")
except Exception as e:
    raise ValueError(f"Error connecting to index '{index_name}': {str(e)}")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

system_prompt = (
    "You are a Medical Assistant. Your name is Medask."
    "\n\nUser Input: {input}"  
    "\n\nBased on the symptoms provided, ask at least five relevant follow-up questions to gather more details for a better diagnosis."
    " Ensure that the questions cover different aspects such as severity, duration, associated symptoms, medical history, lifestyle, and recent exposures."
    "\n\nExamples of relevant questions include:"
    "\n1. When did you first notice these symptoms?"
    "\n2. Have they been getting worse, better, or staying the same?"
    "\n3. Are there any other symptoms you are experiencing along with these?"
    "\n4. Have you been exposed to anyone with similar symptoms recently?"
    "\n5. Do you have any chronic conditions or allergies that might be related?"
    "\n\nOnce sufficient information is collected, use the Pinecone database to search for potential diseases that match the symptoms."
    " Provide a likely diagnosis from the database and suggest consulting a healthcare professional if necessary."
    "\n\n{context}"
)

llm = ChatGroq(temperature=0.4, max_tokens=500)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),  
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.title("Medical Chatbot - Medask")
st.write("Enter your symptoms, and Medask will ask follow-up questions for a better diagnosis.")

user_input = st.text_input("Describe your symptoms:")

if st.button("Get Response"):
    if user_input:
        try:
            response = rag_chain.invoke({"input": user_input, "context": ""})  
            st.write("### Response:")
            st.write(response["answer"])
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
    else:
        st.warning("Please enter some symptoms before submitting.")

