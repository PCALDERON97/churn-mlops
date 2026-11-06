"""Servicio de inferencia: predice si un cliente se va (churn).

Uso local:  uvicorn src.app:app --reload
- Si existe modelo_servido/ (dentro de la imagen Docker) usa ese modelo, que se exportó
  del registro con el alias 'produccion' (ver src/exportar_modelo.py).
- Si no existe, carga directo del registro de MLflow: models:/churn-model@produccion.
Cada respuesta incluye el nombre del pod que la atendió, para ver el balanceo en Kubernetes.
"""
import json
import os
import socket
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent.parent
MODELO_SERVIDO = RAIZ / "modelo_servido"
POD = os.getenv("POD_NAME", socket.gethostname())               # Kubernetes lo inyecta en el Deployment

SiNo = Literal["Yes", "No"]


class Cliente(BaseModel):
    """Las 19 variables de entrada, con los mismos valores que el dataset de Kaggle."""
    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: SiNo = "No"
    Partner: SiNo = "No"
    Dependents: SiNo = "No"
    tenure: float = Field(12, ge=0, le=100, description="meses como cliente")
    PhoneService: SiNo = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: SiNo = "Yes"
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"] = "Electronic check"
    MonthlyCharges: float = Field(80.0, ge=0, le=200)
    TotalCharges: float = Field(960.0, ge=0)


def cargar_modelo():
    """Devuelve (modelo, info del registro)."""
    if MODELO_SERVIDO.exists():
        info = json.loads((MODELO_SERVIDO / "info.json").read_text())
        return mlflow.sklearn.load_model(str(MODELO_SERVIDO / "modelo")), info
    from mlflow.tracking import MlflowClient
    from src.config import ALIAS_PRODUCCION, NOMBRE_MODELO, TRACKING_URI
    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}"
    v = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, ALIAS_PRODUCCION)
    info = {"modelo": NOMBRE_MODELO, "alias": ALIAS_PRODUCCION, "uri": uri, "version": v.version,
            "run_id": v.run_id, "run_name": v.tags.get("run_name"), "f1_test": v.tags.get("f1")}
    return mlflow.sklearn.load_model(uri), info


MODELO, INFO = cargar_modelo()                                   # se carga una sola vez al arrancar
app = FastAPI(title="churn-mlops")


@app.get("/")
def frontend():
    """Página web para probar la API (puntos extra)."""
    return FileResponse(RAIZ / "frontend" / "index.html")


@app.get("/health")
def health():
    """Lo usan las probes de Kubernetes."""
    return {"estado": "ok", "pod": POD}


@app.get("/info")
def info():
    """Qué modelo está sirviendo este pod: versión del registro y run de MLflow."""
    return {"pod": POD, **INFO}


@app.post("/predict")
def predict(cliente: Cliente):
    """Probabilidad de que el cliente se vaya."""
    fila = pd.DataFrame([cliente.model_dump()])
    prob = float(MODELO.predict_proba(fila)[0, 1])
    return {
        "churn": prob >= 0.5,
        "probabilidad": round(prob, 4),
        "pod": POD,
        "modelo_version": INFO["version"],
    }
