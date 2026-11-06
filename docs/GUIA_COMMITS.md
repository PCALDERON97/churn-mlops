# Guía de commits

El historial de Git se revisa como evidencia del trabajo. Conviene ir commiteando por componente, a medida que revisás y probás cada parte, con la convención del curso:

| Prefijo | Para qué |
|---|---|
| `feat:` | nueva funcionalidad |
| `fix:` | corrección de errores |
| `chore:` | configuración, dependencias, mantenimiento |
| `data:` | cambios en datos |
| `exp:` | experimentos de modelos |
| `docs:` | documentación |

## Orden sugerido

```powershell
git init
git branch -M main

git add .gitignore .dockerignore requirements.txt requirements-api.txt pytest.ini
git commit -m "chore: configura dependencias fijadas y archivos ignorados"

git add data/telco_churn.csv src/__init__.py src/config.py src/datos.py tests/__init__.py tests/test_datos.py
git commit -m "data: agrega dataset Telco Churn con limpieza y división estratificada"

git add src/train.py src/promover.py
git commit -m "exp: registra 6 runs en MLflow y versiona el mejor modelo con alias produccion"

git add src/exportar_modelo.py src/app.py tests/test_api.py
git commit -m "feat: agrega API de inferencia que consume models:/churn-model@produccion"

git add Dockerfile
git commit -m "feat: agrega Dockerfile del servicio de inferencia"

git add k8s/ scripts/
git commit -m "feat: despliega la API en Kubernetes con 3 réplicas y Service NodePort"

git add src/drift/__init__.py src/drift/generar_lotes.py src/drift/data_drift.py tests/test_data_drift.py data/lotes/
git commit -m "feat: agrega detección de data drift con KS y PSI"

git add src/drift/concept_drift.py tests/test_concept_drift.py
git commit -m "feat: simula concept drift y define criterio de reentrenamiento"

git add frontend/
git commit -m "feat: agrega interfaz web que consume la API del clúster"

git add README.md docs/
git commit -m "docs: agrega README, arquitectura y evidencias"
```

## Subir a GitHub

Crear el repositorio vacío `churn-mlops` en GitHub y después:

```powershell
git remote add origin https://github.com/<tu-usuario>/churn-mlops.git
git push -u origin main
```

Si para probar algo usás una rama (por ejemplo, un modelo nuevo), integrarla con `git merge` y marcar la versión con un tag: `git tag -a v0.2.0 -m "modelo v2 en producción"`.
