import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
  - 신용 사용률: {CUR}만원

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

다른 어떤 출력도 하지 말고 아래 형식으로만 출력해:

[
  {{
    result: "{{사용자 이름}}님의 신용등급은 {{등급}}입니다."
    tip_1: "수치를 보고 너가 팁을 한 줄 적어"
    tip_2: "하나 더"
  }}
]
팁은 길어도 되지만 최대 50자 이내로.
다시 말한다. 다른 출력은 모두 생략한다.
계산식, 다른 것 다 필요없어. 위의 양식으로만 출력한다
"""

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    response = json.loads(response.choices[0].message.content)
    return response.choices[0].message.content
