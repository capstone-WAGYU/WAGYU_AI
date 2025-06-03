from pydantic import BaseModel

class InvestRequest(BaseModel):
    bank_1: str
    bank_2: str
