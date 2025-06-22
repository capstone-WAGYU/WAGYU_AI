from app.schemas import ProductRequest
from app.schemas import GradeRequest
from app.schemas import InvestRequest
from app.db import save_result_to_db
from app.prodrecom_rag import build_rag_system
from app.prodrecom_utils import json_parse
from app.credit_rating_prompt import getcreditGrade
from app.chatbot_prompt import chatbot_prompt
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
    query = prompt_template.format(bank_1=req.bank_1, bank_2=req.bank_2)
    res = qa.invoke(query)
    parsed_json = json_parse(res["result"])

    if "error" not in parsed_json:
        save_result_to_db(parsed_json)

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