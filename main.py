from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from standard_ai.v1_server.app.schemas import InvestRequest
from standard_ai.v1_server.app.db import save_result_to_db
from standard_ai.v1_server.app.rag import build_rag_system
from standard_ai.v1_server.app.prompt import make_prompt
from standard_ai.v1_server.app.utils import json_parse

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
