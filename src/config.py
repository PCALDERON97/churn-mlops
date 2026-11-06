"""Constantes del proyecto: rutas, columnas y nombres de MLflow."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent          # carpeta del proyecto
DATOS = RAIZ / "data" / "telco_churn.csv"              # dataset original (Kaggle: Telco Customer Churn)
LOTES = RAIZ / "data" / "lotes"                        # lotes nuevos para las pruebas de drift
FIGURAS = RAIZ / "docs" / "figuras"                    # gráficas para la documentación

SEMILLA = 42                                           # semilla fija: mismos resultados en cada corrida
TEST_SIZE = 0.2                                        # 80 % entrenamiento, 20 % prueba (estratificado)

OBJETIVO = "Churn"                                     # 1 = el cliente se fue, 0 = se quedó
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
VARIABLES = NUMERICAS + CATEGORICAS                    # las 19 variables de entrada

# MLflow: dónde se guardan los experimentos y el registro de modelos
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
EXPERIMENTO = "churn-entrenamiento"
EXPERIMENTO_MONITOREO = "churn-monitoreo"
NOMBRE_MODELO = "churn-model"                          # nombre estable en el Model Registry
ALIAS_PRODUCCION = "produccion"                        # la versión que se despliega
METRICA_PRINCIPAL = "f1"                               # F1 de la clase churn (las clases están desbalanceadas)
