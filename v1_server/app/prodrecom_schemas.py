from pydantic import BaseModel

class InvestRequest(BaseModel):
    bank_1: str
    bank_2: str

class GradeRequest(BaseModel):
    PH: str
    PH_1: int
    DL: int
    CHL: int
    CAF: int
    NCA: int
    CUR: float