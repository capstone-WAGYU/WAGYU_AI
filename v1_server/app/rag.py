from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def build_rag_system():
    with open("rag/은행상품_설명.txt", "r", encoding="utf-8") as f:
        bank_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([bank_text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    llm = ChatOpenAI(model="gpt-4o", openai_api_key= openai_api_key, temperature=1.0)

    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
