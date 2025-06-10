from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import InvestRequest
from app.db import save_result_to_db
from app.rag import build_rag_system
from app.prompt import make_prompt
from app.utils import json_parse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa = build_rag_system()
# 라우팅
@app.post("/invrecom", status_code=201)
async def invest_recommend(req: InvestRequest):
    query = make_prompt(req.bank_1, req.bank_2)
    res = qa.invoke(query)
    parsed_json = json_parse(res["result"])
    if "error" not in parsed_json:
        save_result_to_db(parsed_json)
    return parsed_json

# 챗봇

from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pymongo import MongoClient
import datetime
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.environ.get("OPENAI_API_KEY"))
prompt = ChatPromptTemplate.from_template(f"""
너는 개인 자산 관리를 도와주는 전문가 챗봇이야.
너는 세금 관련 질문에 대답할 수 있어.
너는 한국어로 대답해야 해.
아래 Question을 보고, 사용자가 원하는 답변을 해줘.

Question: {{question}}
""")

tax_chain = prompt|llm|StrOutputParser()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Mongourl= os.environ.get("MONGODB_URL")
client = MongoClient(Mongourl)
db = client["chat"]
collection = db["chat_log"]

class QueryInput(BaseModel):
    question: str
@app.get("/chat")
async def get_chat():
    messages = collection.find().sort("time",1)
    chat_history = []
    for message in messages:
        chat_history.append({
            "role": "user",
            "content": message['question']
        })
        chat_history.append({
            "role": "assistant",
            "content": message['answer']
        })
    return chat_history
@app.post("/ask")
async def ask_tax(query: QueryInput):
    result = tax_chain.invoke({"question": query.question})

    timestamp = datetime.datetime.now()

    collection.insert_one({
        "question": query.question,
        "answer": result,
        "time": timestamp
    })
    return {"response": result}