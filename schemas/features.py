from pydantic import BaseModel

class ModelFeatures(BaseModel):
    """Model features"""
    person_age: int	
    person_income: float	
    person_emp_length: float
    loan_intent: str
    loan_grade:	str
    loan_amnt:	float
    loan_int_rate:	float
    loan_percent_income: float
    cb_person_cred_hist_length: float

    