from app.schemas import ProductRequest
from app.schemas import GradeRequest
from app.schemas import InvestRequest
from app.schemas import QueryInput
from app.prodrecom_rag import build_rag_system
from app.prodrecom_utils import json_parse
from app.credit_rating_prompt import getcreditGrade
from app.chatbot_prompt import chatbot_prompt
from app.chatbot_keyword import get_key
from app.invest_recommend import invest
from app.prodrecom_fewshot import get_product_prompt_template
from fastapi import FastAPI
from pydantic import BaseModel
import os
from datetime import date
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
import datetime
from datetime import date
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa = build_rag_system()
prompt_template = get_product_prompt_template()

@app.get("/")
def root():
    return {"Message": "This isn't Error. Rewrite your URL like 'localhost:8000/docs'", 
            "AAA": "Swagger로 가야 뭐든하지병딱아 localhost:8000/docs"}

# 금융상품 추천
@app.post("/prodRecom", status_code=201)
async def invest_recommend(req: ProductRequest):
    query = prompt_template.format(period=req.period, bank=req.bank, country = req.country)
    res = qa.invoke(query)
    parsed_json = json_parse(res["result"])

    return parsed_json

# 투자 추천
@app.post("/invRecom", status_code = 201)
def invest_recommend(data: InvestRequest):
    recommendation = invest(data.interestRate, data.startDate, data.endDate, data.principalGuarantee)
    return recommendation

# 신용등급
@app.post("/getGrade", status_code=201)
async def get_grade(data: GradeRequest):
    result = await getcreditGrade(data.PH, data.PH_1, data.DL, data.CHL, data.CAF, data.NCA, data.CUR)
    return result


# 챗봇

llm = ChatOpenAI(model='gpt-4o-mini', api_key=os.environ.get("OPENAI_API_KEY"))
tax_chain = chatbot_prompt | llm | StrOutputParser()
Mongourl= os.environ.get("MONGO_DB_URL")
client = MongoClient(Mongourl)
db = client["chat"]
collection = db["chat_log"]
messages = collection.find().sort("time",1)

@app.get("/chat")
async def get_chat():
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

@app.get("/key")
async def get_keyword():
    key = get_key(messages)
    key_list = []
    for kw in key:
        key_list.append({
            "keyword" : kw
        })
    return key_list