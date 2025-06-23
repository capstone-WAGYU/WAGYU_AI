import openai
import json
import os
from datetime import date
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 

def invest(interestRate: float, startDate: date, endDate: date, principalGuarantee: str) -> dict:
    prompt = f"""
        너는 투자 추천 봇이다.
        사용자의 수익율, 투자 기간, 원금 보장 여부를 따져 예금, 적금, 펀드, 주식 상품들을 알려주고 팁을 주는 것이 목표다.
        상품들은 실제로 있는 상품들을 추천해줘야 한다.
        다음 사용자의 정보는 이렇다.
        수익률: {interestRate}%, << 가장 중요한 입력이고, 이 정보를 기반으로 어떤 투자 방식을 추천하는지, 또 이유를 작성해
        투자기간: {startDate} ~ {endDate}, << 수익율을 뒷받침해주는 느낌으로
        원금 보장 여부: {principalGuarantee} << 사용자의 희망 사항이야. 

        가중치를 정하자면
        수익율 : 50%
        투자 기간 : 40%
        원금보장여부 : 10%
        이 세 가지를 종합적으로 고려해서 추천 여부를 판단해줘.

        출력은 아래 JSON 형식으로만 해줘. 어디 틀에 씌우지 말고 딱 저렇게만 출력해
        [
        
            {{
                "deposit_reason": "예금 추천 이유를 최소 100자로",
                "deposit_keyword": "추천하는 이유 3가지를 키워드로",
                "savings_reason": "적금 추천 이유를 최소 100자로",
                "savings_keyword": "추천하는 이유 3가지를 키워드로",
                "fund_reason": "펀드 추천 이유를 최소 100자로",
                "fund_keyword": "추천하는 이유 3가지를 키워드로",
                "stock_reason": "주식 추천 이유를 최소 100자로",
                "stock_keyword": "추천하는 이유 3가지를 키워드로",
            }}
        ]
        하나 이상은 추천해야 한다
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
                  {"role": "system", "content": "넌 오직 JSON만 출력하는 투자 추천 봇이다. JSON 외 텍스트는 절대 쓰지 마라."},
                  {"role": "user", "content": prompt}
                ]
    )

    

    import re
    raw_text = response.choices[0].message.content.strip()
    try:
        json_match = re.search(r'\[\s*{.*?}\s*]', raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("응답에서 JSON 블록을 찾을 수 없음.")
        json_str = json_match.group(0)
        result = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n원본:\n{raw_text}")

    return result