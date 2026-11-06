"""Entrena varios modelos, los registra en MLflow y marca el mejor como 'produccion'.

Uso: python -m src.train
- Cada combinación de hiperparámetros es un run del experimento 'churn-entrenamiento'.
- Los 2 mejores runs (por F1) se registran como versiones de 'churn-model'.
- El mejor recibe el alias 'produccion'; el otro, el alias 'candidato'.
"""
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (ALIAS_PRODUCCION, CATEGORICAS, EXPERIMENTO, METRICA_PRINCIPAL, NOMBRE_MODELO,
                        NUMERICAS, SEMILLA, TEST_SIZE, TRACKING_URI)
from src.datos import cargar, dividir

# Las ejecuciones que se comparan: (nombre del run, clase del modelo, hiperparámetros)
EJECUCIONES = [
    ("logreg_C0.1", LogisticRegression, {"C": 0.1, "class_weight": "balanced", "max_iter": 1000}),
    ("logreg_C1", LogisticRegression, {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}),
    ("logreg_C10", LogisticRegression, {"C": 10.0, "class_weight": "balanced", "max_iter": 1000}),
    ("rf_100_d5", RandomForestClassifier, {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced"}),
    ("rf_300_d10", RandomForestClassifier, {"n_estimators": 300, "max_depth": 10, "class_weight": "balanced"}),
    ("gboost_100_d3", GradientBoostingClassifier, {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}),
]


def construir(clase, params):
    """Pipeline completo: preprocesamiento + modelo. Se guarda todo junto en MLflow."""
    preprocesar = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),                               # escala las numéricas
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),       # one-hot de las categóricas
    ])
    if clase is not LogisticRegression:
        params = {**params, "random_state": SEMILLA}
    return Pipeline([("preprocesar", preprocesar), ("modelo", clase(**params))])


def evaluar(modelo, X, y):
    """Métricas sobre el conjunto de prueba (clientes que el modelo no vio)."""
    pred = modelo.predict(X)
    prob = modelo.predict_proba(X)[:, 1]
    return {
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)
    X_train, X_test, y_train, y_test = dividir(cargar())

    resultados = []                                                          # (f1, run_id, nombre)
    for nombre, clase, params in EJECUCIONES:
        with mlflow.start_run(run_name=nombre) as run:
            modelo = construir(clase, params).fit(X_train, y_train)
            metricas = evaluar(modelo, X_test, y_test)

            mlflow.log_params({"modelo": clase.__name__, **params})
            mlflow.log_params({"semilla": SEMILLA, "test_size": TEST_SIZE,
                               "filas_train": len(X_train), "filas_test": len(X_test)})
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(
                modelo, name="modelo",
                signature=infer_signature(X_test, modelo.predict(X_test)),  # esquema de entrada y salida
                input_example=X_test.head(2),
                # los árboles los entrenamos nosotros: se declara ese tipo como confiable para skops
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            resultados.append((metricas[METRICA_PRINCIPAL], run.info.run_id, nombre))
            print(f"{nombre:15s} f1={metricas['f1']:.4f} roc_auc={metricas['roc_auc']:.4f}")

    # Registro: se registran los 2 mejores; el de mayor F1 se registra último
    resultados.sort()
    cliente = MlflowClient()
    versiones = {}
    for f1, run_id, nombre in resultados[-2:]:
        version = mlflow.register_model(f"runs:/{run_id}/modelo", NOMBRE_MODELO)
        cliente.set_model_version_tag(NOMBRE_MODELO, version.version, "run_name", nombre)
        cliente.set_model_version_tag(NOMBRE_MODELO, version.version, "f1", f"{f1:.4f}")
        versiones[nombre] = version.version
        print(f"Registrado {NOMBRE_MODELO} v{version.version} <- run {nombre} ({run_id})")

    mejor = resultados[-1][2]
    segundo = resultados[-2][2]
    cliente.set_registered_model_alias(NOMBRE_MODELO, ALIAS_PRODUCCION, versiones[mejor])
    cliente.set_registered_model_alias(NOMBRE_MODELO, "candidato", versiones[segundo])
    print(f"Alias '{ALIAS_PRODUCCION}' -> v{versiones[mejor]} ({mejor}); 'candidato' -> v{versiones[segundo]} ({segundo})")


if __name__ == "__main__":
    main()
