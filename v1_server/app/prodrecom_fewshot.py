from langchain.prompts import FewShotPromptTemplate, PromptTemplate

def get_product_prompt_template():
    examples = [
        {
            "period": "1년",
            "bank": "KB국민은행",
            "country": "대한민국",
            "output": '''{{
  "product_name_1": "KB국민첫재테크통장",
  "interest_rate_1": "3.2%",
  "product_reason_1": "사회초년생을 위한 자동이체 우대금리로 목돈 마련에 유리함",
  "init_term_1": "만 18세 이상, 국내 거주자",
  "period_1": "1년 자동연장 가능",
  "product_link_1": "https://kb.bank/product/first_savings",

  "product_name2": "KB청년희망적금"
  "interest_rate_2": "기본 연 2.5% + 우대금리 최대 4.5%"
  "product_reason_2": 목돈 마련을 위한 높은 금리 혜택 제공
가입 조건: 19세 이상 34세 이하인 사람
상품 링크 : https://obank.kbstar.com/quics?page=C016613&cc=b061496:b061645&isNew=Y&prcode=DP01001510

  "product_name_3": "신한첫월급우대적금",
  "interest_rate_3": "3.5%",
  "product_reason_3": "첫 급여 입금 시 우대이율 적용으로 초기 자산 형성에 적합",
  "init_term_3": "신한은행 급여계좌 보유자",
  "period_3": "1년 또는 2년",
  "product_link_3": "https://shinhan.com/product/first_salary"
}}'''
        }
    ]

    example_prompt = PromptTemplate(
        input_variables=["period", "bank", "country"],
        template="""
[입력]
기간 : {period},
은행 : {bank},
지역 : {country}

[출력]
{output}
"""
    )

    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""
너는 금융상품을 추천하는 전문가다.
사용자의 정보를 받아 세 개의 금융상품을 추천해야 한다.
사용자에게로부터 오는 정보는 3개다.
출력은 반드시 아래 예시처럼 JSON만 출력하고, 그 외의 문장은 금지한다.
""",
        suffix="""
[입력]
기간 : {period},
은행 : {bank},
지역 : {country}

[출력]
""",
        input_variables=["period", "bank", "country"]
    )
