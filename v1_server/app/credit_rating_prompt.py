# main.py
from fastapi import FastAPI
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

app = FastAPI()

# GPT 클라이언트
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def calculateScore(PH, DL, CHL, CAF, NCA, CUR):
    return (
        PH * 35 +
        DL * 25 +
        CHL * 15 +
        CAF * 10 +
        NCA * 5 +
        CUR * 10
    )

def classifyGrade(score):
    if score >= 801:
        return "1등급"
    elif score >= 651:
        return "2등급"
    elif score >= 501:
        return "3등급"
    elif score >= 351:
        return "4등급"
    else:
        return "5등급"

async def getcreditGrade(PH, PH_1, DL, CHL, CAF, NCA, CUR):
    prompt = f"""
너는 신용등급 판별기이다.

다음 입력값을 바탕으로 아래 수식에 따라 신용점수를 계산하고, 등급표에 따라 등급을 구한다.

- 입력값:
  - 결제이력: {PH}개월 간 {PH_1}만원
  - 부채수준: {DL}만원
  - 신용이력: {CHL}년
  - 신용거래빈도: {CAF}회 (1년 단위) 
  - 신규신용거래: {NCA}건
  - 신용 사용률: {CUR}%

- 각 요소 가중치:
  - 결제이력: 35
  - 부채수준: 25
  - 신용이력: 15
  - 신용거래빈도: 10
  - 신규신용거래: 5
  - 신용 사용률: 10

- 점수 계산:
  신용점수 = (결제이력×35)+(부채수준×25)+(신용이력×15)+(신용거래빈도×10)+(신규신용거래×5)+(신용 사용률×10)

- 등급 기준:
  801~1000 : 1등급
  651~800  : 2등급
  501~650  : 3등급
  351~500  : 4등급
  350 이하 : 5등급

아래 JSON 형식으로만 출력해. 절대로 다른 설명, 줄바꿈, 코드블럭 포함하지 마:
[
  {{
    "result": "사용자님의 신용등급은 {{등급}}입니다.",
    "tip_1": "팁 한 줄 (최소 80자)",
    "tip_2": "또 다른 팁 한 줄 (최소 80자)"
  }}
]

다른거 출력하지 말라 했다. 하면 GEMINI로 바꿀줄알아 씨발아
"""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"GPT 응답이 JSON이 아님: {e}\n받은 내용: {content}")
