from langchain.prompts import FewShotPromptTemplate, PromptTemplate

def get_product_prompt_template():
    examples = [
        {
            "bank_1": "국민은행",
            "bank_2": "신한은행",
            "output": '''{{
  "product_name_1": "KB국민첫재테크통장",
  "interest_rate_1": "3.2%",
  "product_reason_1": "사회초년생을 위한 자동이체 우대금리로 목돈 마련에 유리함",
  "init_term_1": "만 18세 이상, 국내 거주자",
  "period_1": "1년 자동연장 가능",
  "product_link_1": "https://kb.bank/product/first_savings",

  "product_name_2": "신한첫월급우대적금",
  "interest_rate_2": "3.5%",
  "product_reason_2": "첫 급여 입금 시 우대이율 적용으로 초기 자산 형성에 적합",
  "init_term_2": "신한은행 급여계좌 보유자",
  "period_2": "1년 또는 2년",
  "product_link_2": "https://shinhan.com/product/first_salary"
}}'''
        }
    ]

    example_prompt = PromptTemplate(
        input_variables=["bank_1", "bank_2", "output"],
        template="""
[입력]
첫 번째 은행: {bank_1}
두 번째 은행: {bank_2}

[출력]
{output}
"""
    )

    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""
너는 투자 추천 봇이다.
사용자가 선택한 은행 2가지의 상품을 비교하여 사회초년생에게 더 맞는 상품을 추천하라.
각 은행은 실제로 있는 은행들이고, 너도 실제 상품을 추천해야 한다.
각 상품들의 장단점을 비교하고 JSON 형식으로 출력해.
출력은 반드시 아래 예시처럼 JSON만 출력하고, 그 외의 문장은 금지한다.
""",
        suffix="""
[입력]
첫 번째 은행: {bank_1}
두 번째 은행: {bank_2}

[출력]
""",
        input_variables=["bank_1", "bank_2"]
    )
