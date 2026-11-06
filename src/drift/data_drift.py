"""Data drift: compara cada variable de un lote nuevo contra el baseline de entrenamiento.

Uso: python -m src.drift.data_drift <lote.csv>
Termina con código 0 (VERDE) si no hay drift y con código 1 (ROJO) si lo hay.

Pruebas por tipo de variable:
- Numéricas  -> Kolmogórov-Smirnov de dos muestras. Compara las distribuciones completas
                (no solo la media) y no supone normalidad. Alerta si p < 0.05 / 19 (Bonferroni):
                se hacen 19 pruebas a la vez y sin la corrección aparecerían falsas alarmas.
- Categóricas -> PSI (índice de estabilidad poblacional). Mide cuánto cambió la proporción
                de cada categoría. Umbral 0.2: la regla usada en la industria (riesgo crediticio)
                es < 0.1 estable, 0.1-0.2 vigilar, > 0.2 cambio significativo.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import CATEGORICAS, FIGURAS, NUMERICAS, VARIABLES
from src.datos import cargar, dividir

ALFA_KS = 0.05 / len(VARIABLES)      # 0.0026: corrección de Bonferroni por 19 pruebas
UMBRAL_PSI = 0.2


def psi(referencia, actual, eps=1e-4):
    """Índice de estabilidad poblacional entre dos columnas categóricas."""
    categorias = sorted(set(referencia) | set(actual))
    esperado = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy() + eps
    observado = actual.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy() + eps
    return float(np.sum((observado - esperado) * np.log(observado / esperado)))


def detectar(referencia, lote):
    """Devuelve una tabla con una fila por variable y si hay drift o no."""
    filas = []
    for col in NUMERICAS:
        r = ks_2samp(referencia[col], lote[col])
        filas.append({"variable": col, "tipo": "numérica", "prueba": "KS",
                      "valor": r.pvalue, "umbral": f"p < {ALFA_KS:.4f}", "drift": r.pvalue < ALFA_KS})
    for col in CATEGORICAS:
        valor = psi(referencia[col].astype(str), lote[col].astype(str))
        filas.append({"variable": col, "tipo": "categórica", "prueba": "PSI",
                      "valor": valor, "umbral": f"PSI > {UMBRAL_PSI}", "drift": valor > UMBRAL_PSI})
    return pd.DataFrame(filas)


def baseline():
    """Las variables de entrenamiento: contra esto se compara todo lote nuevo."""
    X_train, _, _, _ = dividir(cargar())
    return X_train


def graficar(referencia, lote, nombre):
    """Histogramas de las numéricas: entrenamiento vs lote."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, len(NUMERICAS), figsize=(13, 3.5))
    for eje, col in zip(ejes, NUMERICAS):
        eje.hist(referencia[col], bins=30, alpha=0.55, density=True, label="entrenamiento")
        eje.hist(lote[col], bins=30, alpha=0.55, density=True, label=nombre)
        eje.set_title(col)
    ejes[0].legend()
    fig.suptitle(f"Data drift: entrenamiento vs {nombre}")
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / f"data_drift_{nombre}.png"
    fig.savefig(ruta, dpi=120)
    return ruta


if __name__ == "__main__":
    archivo = sys.argv[1] if len(sys.argv) > 1 else "data/lotes/lote_mismo_origen.csv"
    lote = cargar(archivo)
    ref = baseline()
    tabla = detectar(ref, lote)
    pd.set_option("display.width", 120)
    print(tabla.assign(valor=tabla["valor"].map(lambda v: f"{v:.4g}")).to_string(index=False))
    nombre = archivo.replace("\\", "/").split("/")[-1].removesuffix(".csv")
    print(f"\nGráfica: {graficar(ref, lote, nombre)}")
    con_drift = tabla.loc[tabla["drift"], "variable"].tolist()
    if con_drift:
        print(f"\nESTADO: ROJO - data drift en {len(con_drift)} variable(s): {', '.join(con_drift)}")
        sys.exit(1)
    print("\nESTADO: VERDE - el lote tiene la misma distribución que el entrenamiento")
