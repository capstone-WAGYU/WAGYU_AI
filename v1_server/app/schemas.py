from pydantic import BaseModel
from datetime import date
class ProductRequest(BaseModel):
    period: str
    bank: str
    country: str

class GradeRequest(BaseModel):
    PH: str
    PH_1: int
    DL: int
    CHL: int
    CAF: int
    NCA: int
    CUR: float

class InvestRequest(BaseModel):
    interestRate: float
    startDate: date
    endDate: date
    principalGuarantee: str

class QueryInput(BaseModel):
    question: str