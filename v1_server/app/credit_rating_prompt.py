import openai
from dotenv import load_dotenv
import os
load_dotenv()
client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
def getGrade(PH, PH_1, DL, CHL, CAF, NCA, CUR):
    prompt = f"""
너는 신용등급 판별기이다.

다음 입력값을 바탕으로 아래 수식에 따라 신용점수를 계산하고, 등급표에 따라 등급을 구한다.

- 입력값:
  - 결제이력: {PH}개월 간 {PH_1}만원
  - 부채수준: {DL}만원
  - 신용이력: {CHL}만원
  - 신용거래빈도: {CAF}만원 
  - 신규신용거래: {NCA}만원
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

"{{사용자 이름}}님의 신용등급은 {{등급}}입니다."
"그리고 너만의 팁 2줄."

다시 말한다. 다른 출력은 모두 생략한다.
계산식, 다른 것 다 필요없어. 위의 양식으로만 출력한다
"""


    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content