"""Calibra los umbrales de drift con lotes normales (sin ningún cambio inyectado).

Uso: python -m src.drift.calibrar_umbrales
Responde dos preguntas que justifican los umbrales del documento de arquitectura:

1) Data drift: con lotes que vienen de la misma población que el entrenamiento,
   ¿cuántas veces el detector da una falsa alarma? (se espera casi nunca)
2) Concept drift: si nada cambia, ¿cuánto varía el F1 de un lote a otro solo por azar?
   Con eso se comprueba que una caída del 10 % durante 3 lotes seguidos no pasa por casualidad.
"""
import mlflow
import numpy as np
from sklearn.metrics import f1_score

from src.config import ALIAS_PRODUCCION, FIGURAS, NOMBRE_MODELO, TRACKING_URI
from src.datos import cargar, dividir
from src.drift.concept_drift import CAIDA_MAXIMA, LOTES_SEGUIDOS, TAMANO_LOTE
from src.drift.data_drift import ALFA_KS, UMBRAL_PSI, detectar
from src.drift.generar_lotes import lote_mismo_origen

N_LOTES_DATA = 200       # lotes normales para medir falsas alarmas de data drift
N_LOTES_F1 = 300         # lotes normales para medir la variación del F1


def calibrar_data_drift(X_train, X_test, y_test):
    """Corre el detector sobre lotes limpios y cuenta las falsas alarmas."""
    falsas, psi_max, ks_min = 0, [], []
    for semilla in range(N_LOTES_DATA):
        tabla = detectar(X_train, lote_mismo_origen(X_test, y_test, semilla=semilla))
        falsas += int(tabla["drift"].any())
        psi_max.append(tabla.loc[tabla["prueba"] == "PSI", "valor"].max())
        ks_min.append(tabla.loc[tabla["prueba"] == "KS", "valor"].min())
    return falsas, np.percentile(psi_max, 99), np.percentile(ks_min, 1)


def calibrar_f1(modelo, X_test, y_test):
    """F1 de lotes normales, armados igual que en la simulación de concept drift."""
    rng = np.random.default_rng(0)
    f1s = []
    for _ in range(N_LOTES_F1):
        idx = rng.choice(len(X_test), TAMANO_LOTE, replace=False)
        f1s.append(f1_score(y_test.iloc[idx], modelo.predict(X_test.iloc[idx])))
    return np.array(f1s)


def graficar(f1s, f1_ref, umbral):
    """Histograma del F1 en lotes normales con el umbral de alarma."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, eje = plt.subplots(figsize=(8, 4))
    eje.hist(f1s, bins=30, color="tab:blue", alpha=0.7)
    eje.axvline(f1_ref, color="gray", linestyle="--", label=f"F1 de referencia ({f1_ref:.3f})")
    eje.axvline(umbral, color="tab:red", linestyle=":", label=f"umbral -10 % ({umbral:.3f})")
    eje.set_xlabel("F1 de un lote normal")
    eje.set_ylabel("cantidad de lotes")
    eje.set_title(f"Variación del F1 por azar ({N_LOTES_F1} lotes sin drift)")
    eje.legend()
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / "calibracion_f1.png"
    fig.savefig(ruta, dpi=120)
    return ruta


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}")
    X_train, X_test, _, y_test = dividir(cargar())

    print("=== 1) Data drift con lotes normales ===")
    falsas, psi_p99, ks_p1 = calibrar_data_drift(X_train, X_test, y_test)
    print(f"Lotes revisados:            {N_LOTES_DATA}")
    print(f"Falsas alarmas:             {falsas} ({falsas / N_LOTES_DATA:.1%})")
    print(f"PSI máximo (percentil 99):  {psi_p99:.4f}   umbral de alarma {UMBRAL_PSI}")
    print(f"p-valor KS mínimo (perc. 1): {ks_p1:.4f}   umbral de alarma {ALFA_KS:.4f}")

    print("\n=== 2) Variación del F1 en lotes normales ===")
    f1s = calibrar_f1(modelo, X_test, y_test)
    f1_ref = f1_score(y_test, modelo.predict(X_test))
    umbral = f1_ref * (1 - CAIDA_MAXIMA)
    p_bajo = float((f1s < umbral).mean())
    print(f"Lotes revisados:            {N_LOTES_F1} de {TAMANO_LOTE} clientes")
    print(f"F1 de referencia:           {f1_ref:.3f}")
    print(f"F1 normal: media {f1s.mean():.3f} · desviación {f1s.std():.3f}")
    print(f"Caída del 10 % = {f1_ref - umbral:.3f} = {(f1_ref - umbral) / f1s.std():.1f} desviaciones")
    print(f"Un lote normal cae bajo el umbral:          {p_bajo:.1%}")
    p_seguidos = p_bajo ** LOTES_SEGUIDOS
    texto = f"{p_seguidos:.1e} (aprox. 1 en {1 / p_seguidos:,.0f})" if p_seguidos > 0 else "0 en esta muestra"
    print(f"{LOTES_SEGUIDOS} lotes normales seguidos bajo el umbral: {texto}")
    print(f"\nGráfica: {graficar(f1s, f1_ref, umbral)}")


if __name__ == "__main__":
    main()