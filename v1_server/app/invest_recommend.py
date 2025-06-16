import openai
import json
import os

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 

def invest(return_value, invest_period, guaranteed_principal):
    prompt = f"""
        너는 투자 추천 봇이다.
        사용자의 수익율, 투자 기간, 원금 보장 여부를 따져 예금, 적금, 펀드, 주식 상품들을 알려주고 팁을 주는 것이 목표다.
        상품들은 실제로 있는 상품들을 추천해줘야 한다.
        다음 사용자의 정보는 이렇다.
        수익률: {return_value}%, << 가장 중요한 입력이고, 이 정보를 기반으로 어떤 투자 방식을 추천하는지, 또 이유를 작성해
        투자기간: {invest_period}개월, << 수익율을 뒷받침해주는 느낌으로
        원금 보장 여부: {guaranteed_principal} << 사용자의 희망 사항이야. 

        가중치를 정하자면
        수익율 : 50%
        투자 기간 : 40%
        원금보장여부 : 10%
        이 세 가지를 종합적으로 고려해서 추천 여부를 판단해줘.

        출력은 아래 JSON 형식으로만 해줘.
        [
        
            {{
                "deposit_recommend": "예금 추천여부를 한 문장으로",
                "deposit_reason": "예금 추천 이유를 최소 100자로",
                "deposit_keyword": "추천하는 이유 3가지를 키워드로",
                "savings_recommend": "적금 추천여부를 한 문장으로",
                "savings_reason": "적금 추천 이유를 최소 100자로",
                "savings_keyword": "추천하는 이유 3가지를 키워드로",
                "fund_recommend": "펀드 추천여부를 한 문장으로",
                "fund_reason": "펀드 추천 이유를 최소 100자로",
                "fund_keyword": "추천하는 이유 3가지를 키워드로",
                "stock_recommend": "주식 추천여부를 한 문장으로",
                "stock_reason": "주식 추천 이유를 최소 100자로",
                "stock_keyword": "추천하는 이유 3가지를 키워드로",
            }}
        ]
        하나 이상은 추천해야 한다
        다른 이상한 출력 내놓으면 죽을 줄 알아
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.choices[0].message.content.strip()
    
    # JSON 문자열을 Python 딕셔너리로 파싱
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"응답 파싱 실패: {raw_text}")

    return result
