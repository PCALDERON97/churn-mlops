"""Genera los lotes "nuevos" que llegan a producción, para probar la detección de data drift.

Uso: python -m src.drift.generar_lotes
- lote_mismo_origen.csv: clientes del conjunto de prueba (misma distribución que el entrenamiento).
- lote_derivado.csv: los mismos clientes con cambios inyectados a propósito:
    * llegan clientes más nuevos (tenure a la mitad),
    * suben los precios un 25 % (MonthlyCharges),
    * más contratos mes a mes y más pago con cheque electrónico (campaña agresiva de captación).
"""
import numpy as np

from src.config import LOTES, SEMILLA
from src.datos import cargar, dividir

TAMANO_LOTE = 1000


def lote_mismo_origen(X_test, y_test, semilla=SEMILLA):
    """Muestra aleatoria del conjunto de prueba: no debería haber drift."""
    lote = X_test.sample(TAMANO_LOTE, random_state=semilla).copy()
    lote["Churn"] = y_test.loc[lote.index]
    return lote.reset_index(drop=True)


def inyectar_drift(lote, semilla=SEMILLA):
    """Aplica los cambios de distribución descritos arriba."""
    rng = np.random.default_rng(semilla)
    d = lote.copy()
    d["tenure"] = (d["tenure"] * 0.5).round()                                       # clientes más nuevos
    d["MonthlyCharges"] = (d["MonthlyCharges"] * 1.25).round(2)                     # precios 25 % más altos
    d["TotalCharges"] = (d["tenure"] * d["MonthlyCharges"]).round(2)                # coherente con lo anterior
    cambia = rng.random(len(d)) < 0.35
    d.loc[cambia, "Contract"] = "Month-to-month"                                    # más contratos mensuales
    cambia = rng.random(len(d)) < 0.35
    d.loc[cambia, "PaymentMethod"] = "Electronic check"                             # más pago electrónico
    return d


if __name__ == "__main__":
    LOTES.mkdir(parents=True, exist_ok=True)
    _, X_test, _, y_test = dividir(cargar())
    normal = lote_mismo_origen(X_test, y_test)
    normal.to_csv(LOTES / "lote_mismo_origen.csv", index=False)
    inyectar_drift(normal).to_csv(LOTES / "lote_derivado.csv", index=False)
    print(f"Lotes guardados en {LOTES}: lote_mismo_origen.csv y lote_derivado.csv ({TAMANO_LOTE} clientes cada uno)")
