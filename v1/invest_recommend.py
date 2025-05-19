from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import openai
import json
from fastapi.middleware.cors import CORSMiddleware

openai.api_key = "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"

with open("./rag/은행상품_설명.txt", "r", encoding="utf-8") as f:
    bank_text = f.read()

#문서 스플릿
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([bank_text])

# 임베딩, 벡터스토어
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectorstore = FAISS.from_documents(docs, embeddings)
#모델
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=openai.api_key, temperature=1.0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 개발 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvestRequest(BaseModel):
    bank_1: str
    bank_2: str

@app.post("/invrecom", status_code = 201)
async def invest_recommend(req: InvestRequest):
    bank_1 = req.bank_1
    bank_2 = req.bank_2

    query = f"""
    너는 투자 추천 봇이다.
    사용자가 선택한 은행 2가지의 상품을 비교하여 사회초년생에게 더 맞는 상품을 추천하라.
    각 은행은 실제로 있는 은행들이고, 너도 실제 상품을 추천해야 한다.
    첫 번째 은행은 {bank_1}, 두 번째 은행은 {bank_2}이다.
    각 상품들의 장단점을 비교하고 JSON 형식으로 출력해.

    출력 형식은 아래와 같다:
    [
      {{
        "product_name_1": "첫 번째 은행의 금융상품 이름",
        "product_reason_1": "이율, 안정성, 사회초년생에게 필요한 이유를 짧게 서술",
        "product_name_2": "두 번째 은행의 금융상품 이름",
        "product_reason_2": "이율, 안정성, 사회초년생에게 필요한 이유를 짧게 서술"
      }}
    ]
    product_reason 둘이 느낌이 비슷하면 너의 재량으로 수정해서 차이점을 설명하지만, JSON 형식을 지켜.
    출력 형식 외에는 어떤 텍스트도 포함하지 마.
    다른 형식으로 출력하면 죽는줄알아
    """

    res = qa.invoke(query)
    jsonstr = res["result"]

    try:
        parsed_json = json.loads(jsonstr)
    except json.JSONDecodeError:
        return {
            "error": "JSON 파싱 실패. 응답 형식이 올바르지 않음.",
            "raw": jsonstr
        }

    return parsed_json
