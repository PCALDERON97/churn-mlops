"""Carga, limpieza y división de los datos."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CATEGORICAS, DATOS, NUMERICAS, OBJETIVO, SEMILLA, TEST_SIZE, VARIABLES


def cargar(ruta=DATOS):
    """Lee el CSV y deja cada columna con su tipo correcto."""
    df = pd.read_csv(ruta)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)  # 11 clientes nuevos vienen vacíos
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])  # es categórica
    for col in CATEGORICAS:
        df[col] = df[col].astype(str)
    for col in NUMERICAS:
        df[col] = df[col].astype(float)
    if OBJETIVO in df.columns and not pd.api.types.is_numeric_dtype(df[OBJETIVO]):
        df[OBJETIVO] = (df[OBJETIVO] == "Yes").astype(int)
    return df


def dividir(df):
    """Divide en entrenamiento y prueba de forma reproducible y estratificada por la etiqueta."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[VARIABLES], df[OBJETIVO], test_size=TEST_SIZE, random_state=SEMILLA, stratify=df[OBJETIVO]
    )
    return X_train, X_test, y_train, y_test
