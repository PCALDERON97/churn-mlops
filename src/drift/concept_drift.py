"""Concept drift: la relación entre las variables y el churn cambia y el modelo se degrada.

Uso: python -m src.drift.concept_drift
Simula 12 lotes semanales de clientes con su etiqueta real. Las variables de entrada se sacan
siempre de la misma población (el conjunto de prueba), así que NO hay data drift. Lo que cambia
desde la semana 5 es la regla que genera el churn:

  Un competidor lanza una oferta para contratos largos, y la empresa lanza una tarifa de
  retención para contratos mensuales. Entonces:
    * parte de los clientes mes a mes que antes se iban ahora se quedan,
    * parte de los clientes con contrato anual o de dos años que se quedaban ahora se van.
  La intensidad crece semana a semana (drift gradual).

Criterio de reentrenamiento: alarma si el F1 cae más de un 10 % respecto del F1 de referencia
(el del conjunto de prueba al registrar el modelo) durante 3 lotes seguidos.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import mlflow
from src.config import ALIAS_PRODUCCION, EXPERIMENTO_MONITOREO, FIGURAS, NOMBRE_MODELO, SEMILLA, TRACKING_URI
from src.datos import cargar, dividir
from src.drift.data_drift import detectar

N_LOTES = 12
TAMANO_LOTE = 500
INICIO_DRIFT = 5                 # semana en la que cambia la regla
CAIDA_MAXIMA = 0.10              # 10 % de caída relativa del F1
LOTES_SEGUIDOS = 3               # cuántos lotes seguidos por debajo del umbral disparan la alarma


def intensidad(semana):
    """Qué fracción de los clientes afectados cambia de comportamiento esa semana."""
    return 0.0 if semana < INICIO_DRIFT else min(0.15 * (semana - INICIO_DRIFT + 1), 0.75)


def aplicar_concepto_nuevo(X, y, p, rng):
    """Cambia las etiquetas según la regla nueva. Las variables X no se tocan."""
    y = y.copy()
    mensual = (X["Contract"] == "Month-to-month").to_numpy()
    se_queda = mensual & (y.to_numpy() == 1) & (rng.random(len(y)) < p)                 # la tarifa de retención funciona
    se_va = ~mensual & (y.to_numpy() == 0) & (rng.random(len(y)) < p / 2)               # el competidor se los lleva
    y[se_queda] = 0
    y[se_va] = 1
    return y


def simular(modelo, X_test, y_test, f1_referencia, referencia_X):
    """Recorre los lotes y devuelve una tabla con la métrica y el estado de cada semana."""
    rng = np.random.default_rng(SEMILLA)
    umbral = f1_referencia * (1 - CAIDA_MAXIMA)
    filas, seguidos = [], 0
    for semana in range(1, N_LOTES + 1):
        idx = rng.choice(len(X_test), TAMANO_LOTE, replace=False)                        # clientes de la semana
        X = X_test.iloc[idx].reset_index(drop=True)
        y = aplicar_concepto_nuevo(X, y_test.iloc[idx].reset_index(drop=True), intensidad(semana), rng)
        pred = modelo.predict(X)
        f1 = f1_score(y, pred)
        seguidos = seguidos + 1 if f1 < umbral else 0
        filas.append({
            "semana": semana,
            "intensidad_drift": intensidad(semana),
            "f1": f1,
            "tasa_churn_predicha": float(pred.mean()),          # no necesita etiquetas
            "tasa_churn_real": float(y.mean()),                 # solo se conoce cuando llega la etiqueta
            "data_drift": bool(detectar(referencia_X, X)["drift"].any()),
            "bajo_umbral": f1 < umbral,
            "alarma_reentrenar": seguidos >= LOTES_SEGUIDOS,
        })
    return pd.DataFrame(filas), umbral


def graficar(tabla, f1_referencia, umbral):
    """Gráfica temporal del F1 por lote."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, eje = plt.subplots(figsize=(10, 4.5))
    eje.axvspan(INICIO_DRIFT - 0.5, N_LOTES + 0.5, color="orange", alpha=0.08, label="cambia la regla (concept drift)")
    eje.plot(tabla["semana"], tabla["f1"], marker="o", color="tab:blue", label="F1 del lote")
    eje.axhline(f1_referencia, color="gray", linestyle="--", label=f"F1 de referencia ({f1_referencia:.3f})")
    eje.axhline(umbral, color="tab:red", linestyle=":", label=f"umbral -10 % ({umbral:.3f})")
    alarmas = tabla[tabla["alarma_reentrenar"]]
    eje.scatter(alarmas["semana"], alarmas["f1"], s=160, facecolors="none", edgecolors="tab:red",
                linewidths=2, label="alarma: reentrenar", zorder=5)
    eje.set_xlabel("lote (semana)")
    eje.set_ylabel("F1 (clase churn)")
    eje.set_xticks(tabla["semana"])
    eje.set_title("Concept drift: degradación del modelo en producción")
    eje.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / "concept_drift.png"
    fig.savefig(ruta, dpi=120)
    return ruta


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    from mlflow.tracking import MlflowClient
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, ALIAS_PRODUCCION)
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}")

    X_train, X_test, _, y_test = dividir(cargar())
    f1_referencia = f1_score(y_test, modelo.predict(X_test))
    tabla, umbral = simular(modelo, X_test, y_test, f1_referencia, X_train)
    ruta = graficar(tabla, f1_referencia, umbral)

    # El monitoreo queda en MLflow: cada lote es un paso de la métrica -> gráfica temporal en la UI
    mlflow.set_experiment(EXPERIMENTO_MONITOREO)
    with mlflow.start_run(run_name=f"concept_drift_v{version.version}"):
        mlflow.log_params({"modelo_version": version.version, "modelo_run_id": version.run_id,
                           "caida_maxima": CAIDA_MAXIMA, "lotes_seguidos": LOTES_SEGUIDOS,
                           "inicio_drift": INICIO_DRIFT, "tamano_lote": TAMANO_LOTE})
        for fila in tabla.itertuples():
            mlflow.log_metrics({"f1_lote": fila.f1, "tasa_churn_predicha": fila.tasa_churn_predicha,
                                "tasa_churn_real": fila.tasa_churn_real}, step=fila.semana)
        mlflow.log_artifact(str(ruta))

    pd.set_option("display.width", 140)
    print(tabla.round(3).to_string(index=False))
    print(f"\nF1 de referencia {f1_referencia:.3f} · umbral {umbral:.3f} · gráfica {ruta}")
    primera = tabla.loc[tabla["alarma_reentrenar"], "semana"]
    if len(primera):
        print(f"ALARMA: reentrenar a partir de la semana {int(primera.iloc[0])}")
    else:
        print("Sin alarma: el modelo sigue dentro del criterio")


if __name__ == "__main__":
    main()
