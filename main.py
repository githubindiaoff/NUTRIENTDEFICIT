from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predictor import predict
from report_generator import generate_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClinicalInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: ClinicalInput):
    prediction, confidence = predict(input.text)
    report = generate_report(prediction, confidence)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "report": report
    }