"""El servicio de inferencia responde y dice qué versión del modelo usa."""
from fastapi.testclient import TestClient

from src.app import app

cliente = TestClient(app)


def test_health():
    assert cliente.get("/health").json()["estado"] == "ok"


def test_info_trazabilidad_con_el_registro():
    info = cliente.get("/info").json()
    assert info["uri"] == "models:/churn-model@produccion"
    assert info["version"] and info["run_id"]


def test_predice_una_probabilidad():
    r = cliente.post("/predict", json={"tenure": 2, "Contract": "Month-to-month"}).json()
    assert 0 <= r["probabilidad"] <= 1 and "pod" in r


def test_rechaza_datos_invalidos():
    assert cliente.post("/predict", json={"tenure": -5}).status_code == 422
