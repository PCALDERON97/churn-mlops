# Imagen del servicio de inferencia: API + frontend + modelo exportado del registro de MLflow
FROM python:3.12-slim

WORKDIR /app

# 1) Librerías primero: Docker reutiliza esta capa mientras no cambie requirements-api.txt
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2) Código y modelo. modelo_servido/ se genera con: python -m src.exportar_modelo
COPY src/__init__.py src/config.py src/app.py src/
COPY frontend/ frontend/
COPY modelo_servido/ modelo_servido/

# Usuario sin privilegios (buena práctica de seguridad)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
