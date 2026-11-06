"""Resuelve 'models:/churn-model@produccion' en el registro y lo prepara para la imagen Docker.

Uso: python -m src.exportar_modelo
Deja en modelo_servido/:
- modelo/      el modelo descargado del registro (formato MLflow)
- info.json    nombre, alias, versión y run del registro -> trazabilidad desde el clúster
"""
import json
import shutil

import mlflow
from mlflow.tracking import MlflowClient

from src.config import ALIAS_PRODUCCION, NOMBRE_MODELO, RAIZ, TRACKING_URI

DESTINO = RAIZ / "modelo_servido"

if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}"                       # se pide por alias, no por ruta
    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, ALIAS_PRODUCCION)

    shutil.rmtree(DESTINO, ignore_errors=True)
    DESTINO.mkdir()
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(DESTINO / "modelo"))
    info = {
        "modelo": NOMBRE_MODELO,
        "alias": ALIAS_PRODUCCION,
        "uri": uri,
        "version": version.version,
        "run_id": version.run_id,
        "run_name": version.tags.get("run_name"),
        "f1_test": version.tags.get("f1"),
    }
    (DESTINO / "info.json").write_text(json.dumps(info, indent=2))
    print(f"Exportado {uri} -> versión {version.version}, run {version.run_id}")
