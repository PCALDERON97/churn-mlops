"""Data drift.

1) Pruebas del detector (siempre deben pasar):
   - con datos del mismo origen no hay drift,
   - con datos derivados a propósito sí lo hay.
2) Prueba del lote que llega (test_lote_sin_drift): VERDE o ROJO según el archivo.
   Por defecto revisa el lote del mismo origen (VERDE). Para ver el ROJO:
     Linux/macOS:  LOTE=data/lotes/lote_derivado.csv pytest tests/test_data_drift.py -k lote
     PowerShell:   $env:LOTE="data/lotes/lote_derivado.csv"; pytest tests/test_data_drift.py -k lote
"""
import os

import pytest

from src.config import LOTES
from src.datos import cargar, dividir
from src.drift.data_drift import baseline, detectar
from src.drift.generar_lotes import inyectar_drift, lote_mismo_origen


@pytest.fixture(scope="module")
def datos():
    X_train, X_test, _, y_test = dividir(cargar())
    return X_train, lote_mismo_origen(X_test, y_test)


def test_detector_no_alarma_con_datos_del_mismo_origen(datos):
    referencia, normal = datos
    assert not detectar(referencia, normal)["drift"].any()


def test_detector_alarma_con_datos_derivados(datos):
    referencia, normal = datos
    tabla = detectar(referencia, inyectar_drift(normal))
    assert set(tabla.loc[tabla["drift"], "variable"]) >= {"tenure", "MonthlyCharges"}


def test_lote_sin_drift():
    archivo = os.getenv("LOTE", str(LOTES / "lote_mismo_origen.csv"))
    tabla = detectar(baseline(), cargar(archivo))
    con_drift = tabla.loc[tabla["drift"], "variable"].tolist()
    assert not con_drift, f"DATA DRIFT en {archivo}: {con_drift}"
