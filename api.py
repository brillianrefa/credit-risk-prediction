"""REST API prediksi risiko kredit untuk tim backend.

Jalankan:
    uvicorn api:app --host 0.0.0.0 --port 8000

Dokumentasi interaktif otomatis tersedia di http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional
from database import engine, Base, get_db
import models
from sqlalchemy.orm import Session
from fastapi import Depends
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

import predictor


class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanIntent(str, Enum):
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"
    EDUCATION = "EDUCATION"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    MEDICAL = "MEDICAL"
    PERSONAL = "PERSONAL"
    VENTURE = "VENTURE"


class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class DefaultFlag(str, Enum):
    N = "N"
    Y = "Y"


class ThresholdOverride(BaseModel):
    t_low: float = Field(ge=0.01, le=0.99,
                         description="Batas bawah zona review (disetujui bila PD < t_low)")
    t_high: float = Field(ge=0.01, le=0.99,
                          description="Batas tolak (ditolak bila PD >= t_high)")

    @model_validator(mode="after")
    def cek_urutan(self):
        if self.t_low >= self.t_high:
            raise ValueError("t_low harus lebih kecil dari t_high")
        return self


class LoanApplication(BaseModel):
    person_age: int = Field(ge=18, le=100, examples=[30],
                            description="Usia nasabah (tahun)")
    person_income: float = Field(gt=0, examples=[55000],
                                 description="Pendapatan tahunan (tanpa batas atas)")
    person_home_ownership: HomeOwnership = Field(examples=["RENT"],
                                                 description="Status kepemilikan rumah")
    person_emp_length: Optional[float] = Field(default=None, ge=0, le=60,
                                               examples=[4.0],
                                               description="Lama kerja (tahun); null = tidak diketahui")
    loan_intent: LoanIntent = Field(examples=["PERSONAL"],
                                    description="Tujuan pinjaman")
    loan_grade: LoanGrade = Field(examples=["D"], description="Grade pinjaman A-G")
    loan_amnt: float = Field(gt=0, le=100_000_000, examples=[8000],
                             description="Jumlah pinjaman (maks 100.000.000)")
    loan_int_rate: Optional[float] = Field(default=None, ge=0, le=30,
                                           examples=[11.0],
                                           description="Suku bunga (%); null = tidak diketahui")
    loan_percent_income: float = Field(ge=0, le=1, examples=[0.15],
                                       description="Rasio jumlah pinjaman terhadap pendapatan")
    cb_person_default_on_file: DefaultFlag = Field(default=DefaultFlag.N,
                                                   description="Pernah gagal bayar tercatat (Y/N)")
    cb_person_cred_hist_length: int = Field(ge=0, le=50, examples=[4],
                                            description="Panjang riwayat kredit (tahun), maks 50")
    threshold: Optional[ThresholdOverride] = Field(
        default=None, description="Override threshold; null = pakai nilai default model")


class ThresholdDipakai(BaseModel):
    t_low: float
    t_high: float
    sumber: str


class PredictionResponse(BaseModel):
    probabilitas_gagal_bayar: float = Field(description="PD terkalibrasi (0-1)")
    skor_mentah: float = Field(description="Skor mentah model sebelum kalibrasi")
    keputusan: str = Field(description="DISETUJUI / REVIEW_MANUAL / DITOLAK")
    zona: str = Field(description="approve / review / decline")
    risiko: str = Field(description="RENDAH / SEDANG / TINGGI")
    threshold_dipakai: ThresholdDipakai
    catatan: list[str]
    peringatan: list[str] = Field(description="Peringatan bila input di luar rentang data latih")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor.load_bundle()
        print("[API] Model berhasil dimuat.")
    except Exception as exc:
        print(f"[API] PERINGATAN: model gagal dimuat: {exc}")
    yield


app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0",
    description=(
        "API prediksi risiko gagal bayar nasabah untuk platform kredit. "
        "Model: XGBoost + Z-Score + kalibrasi Platt. "
        "Deliverable model untuk tim backend."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Monitoring"])
def health():
    try:
        predictor.load_bundle()
        return {"status": "ok", "model_loaded": True}
    except Exception as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


@app.get("/model-info", tags=["Model"])
def model_info():
    try:
        return predictor.model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memuat model: {exc}")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediksi"])
def predict(payload: LoanApplication, db: Session = Depends(get_db)):
    data = payload.model_dump(mode="json")
    threshold = data.pop("threshold")
    
    if threshold:
        result = predictor.predict_one(data, t_low=threshold["t_low"], t_high=threshold["t_high"])
    else:
        result = predictor.predict_one(data)
    db_record = models.LoanApplicationRecord(
        person_age=payload.person_age,
        person_income=payload.person_income,
        person_home_ownership=payload.person_home_ownership.value,
        person_emp_length=payload.person_emp_length,
        loan_intent=payload.loan_intent.value,
        loan_grade=payload.loan_grade.value,
        loan_amnt=payload.loan_amnt,
        loan_int_rate=payload.loan_int_rate,
        loan_percent_income=payload.loan_percent_income,
        cb_person_default_on_file=payload.cb_person_default_on_file.value,
        cb_person_cred_hist_length=payload.cb_person_cred_hist_length,
        probabilitas_gagal_bayar=result["probabilitas_gagal_bayar"],
        skor_mentah=result["skor_mentah"],
        keputusan=result["keputusan"],
        zona=result["zona"],
        risiko=result["risiko"],
    )
    db.add(db_record)
    db.commit()      
    db.refresh(db_record)
    return result


@app.get("/history", tags=["Riwayat"])
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    records = db.query(models.LoanApplicationRecord).order_by(
        models.LoanApplicationRecord.created_at.desc()
    ).limit(limit).all()
    return records


Base.metadata.create_all(bind=engine)