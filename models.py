from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base

class LoanApplicationRecord(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    person_age = Column(Integer, nullable=False)
    person_income = Column(Float, nullable=False)
    person_home_ownership = Column(String(20), nullable=False)
    person_emp_length = Column(Float, nullable=True)
    loan_intent = Column(String(50), nullable=False)
    loan_grade = Column(String(5), nullable=False)
    loan_amnt = Column(Float, nullable=False)
    loan_int_rate = Column(Float, nullable=True)
    loan_percent_income = Column(Float, nullable=False)
    cb_person_default_on_file = Column(String(5), nullable=False)
    cb_person_cred_hist_length = Column(Integer, nullable=False)

    probabilitas_gagal_bayar = Column(Float, nullable=False)
    skor_mentah = Column(Float, nullable=False)
    keputusan = Column(String(20), nullable=False)  
    zona = Column(String(20), nullable=False)       
    risiko = Column(String(20), nullable=False)     