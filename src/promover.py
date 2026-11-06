"""Mueve el alias 'produccion' a otra versión del registro (promoción o rollback).

Uso: python -m src.promover <version>
"""
import sys

from mlflow.tracking import MlflowClient

import mlflow
from src.config import ALIAS_PRODUCCION, NOMBRE_MODELO, TRACKING_URI

if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    version = sys.argv[1]
    cliente = MlflowClient()
    cliente.get_model_version(NOMBRE_MODELO, version)                    # falla si la versión no existe
    cliente.set_registered_model_alias(NOMBRE_MODELO, ALIAS_PRODUCCION, version)
    print(f"Alias '{ALIAS_PRODUCCION}' de {NOMBRE_MODELO} -> versión {version}")
