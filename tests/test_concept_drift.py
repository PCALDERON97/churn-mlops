"""Concept drift: el criterio de reentrenamiento salta solo cuando cambia la regla."""
import mlflow
import pytest
from sklearn.metrics import f1_score

from src.config import TRACKING_URI
from src.datos import cargar, dividir
from src.drift.concept_drift import INICIO_DRIFT, simular


@pytest.fixture(scope="module")
def tabla():
    mlflow.set_tracking_uri(TRACKING_URI)
    modelo = mlflow.sklearn.load_model("models:/churn-model@produccion")
    X_train, X_test, _, y_test = dividir(cargar())
    resultado, _ = simular(modelo, X_test, y_test, f1_score(y_test, modelo.predict(X_test)), X_train)
    return resultado


def test_sin_alarma_antes_del_cambio(tabla):
    assert not tabla.loc[tabla["semana"] < INICIO_DRIFT, "alarma_reentrenar"].any()


def test_alarma_despues_del_cambio(tabla):
    assert tabla.loc[tabla["semana"] >= INICIO_DRIFT, "alarma_reentrenar"].any()


def test_las_entradas_no_cambian_es_concept_drift_y_no_data_drift(tabla):
    assert not tabla["data_drift"].any()
