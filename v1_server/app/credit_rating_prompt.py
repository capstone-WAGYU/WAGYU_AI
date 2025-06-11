import openai
from dotenv import load_dotenv
import os
load_dotenv()
client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
def get_credit_grade(name, 결제이력, 부채수준, 신용이력, 신용거래빈도, 신규신용거래):
    prompt = f"""
너는 신용등급 판별기이다.

다음 입력값을 바탕으로 아래 수식에 따라 신용점수를 계산하고, 등급표에 따라 등급을 구한다.

- 입력값:
  - 결제이력: {결제이력}
  - 부채수준: {부채수준}
  - 신용이력: {신용이력}
  - 신용거래빈도: {신용거래빈도}
  - 신규신용거래: {신규신용거래}

- 각 요소 가중치:
  - 결제이력: 35
  - 부채수준: 30
  - 신용이력: 25
  - 신용거래빈도: 10
  - 신규신용거래: 5

- 점수 계산:
  신용점수 = (결제이력×35)+(부채수준×30)+(신용이력×25)+(신용거래빈도×10)+(신규신용거래×5)

- 등급 기준:
  801~1000 : 1등급
  651~800  : 2등급
  501~650  : 3등급
  400~500  : 4등급
  400 미만 : 5등급

다른 어떤 출력도 하지 말고 아래 형식으로만 출력해:

"{name}님의 신용등급은 {{등급}}입니다."

다시 말한다. 다른 출력은 모두 생략한다.
계산식, 다른 것 다 필요없어. 위의 양식으로만 출력한다
"""


    response = client.chat.completions.create(
        model="gpt-4-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content