# churn-mlops

Modelo que predice si un cliente de una telefónica se va a ir (**churn**), llevado desde el entrenamiento hasta un despliegue reproducible y observable:

- **MLflow**: 6 runs comparables, Model Registry con 2 versiones y alias `produccion`.
- **Docker**: imagen propia con dependencias fijadas, que levanta la API sola.
- **Kubernetes**: 3 réplicas detrás de un Service, con autorreparación y escalado.
- **Drift**: pruebas automatizadas de data drift (KS y PSI) y de concept drift (F1 por lote).
- **Extra**: página web que consume la API desplegada en el clúster.

Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7.043 clientes, 19 variables (3 numéricas y 16 categóricas). La copia en `data/telco_churn.csv` es la versión que publica IBM en GitHub, idéntica a la de Kaggle.

La justificación de cada decisión está en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

---

## Estructura

```
churn-mlops/
├── data/
│   ├── telco_churn.csv          # dataset original
│   └── lotes/                   # lotes nuevos para probar data drift
├── src/
│   ├── config.py                # rutas, columnas, nombres de MLflow
│   ├── datos.py                 # carga, limpieza y división train/test
│   ├── train.py                 # 6 runs en MLflow + registro + alias
│   ├── promover.py              # mueve el alias 'produccion' a otra versión
│   ├── exportar_modelo.py       # resuelve models:/churn-model@produccion para la imagen
│   ├── app.py                   # API FastAPI: /predict, /info, /health y la página
│   └── drift/
│       ├── generar_lotes.py     # lote del mismo origen y lote con drift inyectado
│       ├── data_drift.py        # KS (numéricas) y PSI (categóricas) -> VERDE / ROJO
│       └── concept_drift.py     # 12 lotes, la regla cambia, F1 en el tiempo, alarma
├── tests/                       # pytest: datos, API, data drift y concept drift
├── frontend/index.html          # interfaz web (puntos extra)
├── k8s/                         # namespace, deployment (3 réplicas) y service
├── scripts/                     # demos de balanceo y autorreparación
├── docs/                        # arquitectura y figuras
├── Dockerfile
├── requirements.txt             # entrenamiento, drift y pruebas
└── requirements-api.txt         # solo lo que va en la imagen
```

---

## Paso a paso (Windows PowerShell)

Requisitos: Python 3.12, Docker Desktop con Kubernetes activado y Git.

### 1. Entorno

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Fase 1: entrenamiento y MLflow

```powershell
python -m src.train                     # 6 runs, registra v1 y v2, alias 'produccion' al mejor
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir <http://localhost:5000>:

- **Experiments → churn-entrenamiento**: los 6 runs. Ordenar por `f1`, seleccionar varios y usar **Compare**.
- **Models → churn-model**: las 2 versiones, una con el alias `produccion` y otra con `candidato`.

Para cambiar qué versión se despliega (por ejemplo, un rollback):

```powershell
python -m src.promover 1
```

### 3. Fase 2: imagen Docker

```powershell
python -m src.exportar_modelo           # baja models:/churn-model@produccion a modelo_servido/
docker build -t churn-api:v2 .          # el tag = versión del registro (ver modelo_servido/info.json)
docker run --rm -p 8000:8000 churn-api:v2
```

Probar en <http://localhost:8000> (página) o en <http://localhost:8000/docs> (Swagger).

### 4. Fase 3: Kubernetes

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods -n churn -o wide
```

Si la versión de producción no es la 2, cambiar `image: churn-api:v2` en `k8s/deployment.yaml` por el tag correcto.

| Demostración | Comandos |
|---|---|
| 3 réplicas Running | `kubectl get pods -n churn -o wide` |
| Balanceo entre pods | `.\scripts\probar_balanceo.ps1` |
| Autorreparación sin corte | Terminal 1: `.\scripts\vigilar_servicio.ps1` · Terminal 2: `kubectl delete pod <pod> -n churn` y `kubectl get pods -n churn` |
| Escalado | `kubectl scale deployment churn-api -n churn --replicas=5` y luego `--replicas=3` |

> Si PowerShell no deja ejecutar los scripts ("la ejecución de scripts está deshabilitada"): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

La página web desde el clúster: <http://localhost:30080>. Cada predicción muestra qué pod la atendió.

> **minikube** en lugar de Docker Desktop: `minikube image load churn-api:v2` antes de aplicar los manifiestos, y `minikube service churn-api -n churn --url` para obtener la URL.

### 5. Fase 4: drift

```powershell
python -m src.drift.generar_lotes
python -m src.drift.data_drift data/lotes/lote_mismo_origen.csv   # VERDE
python -m src.drift.data_drift data/lotes/lote_derivado.csv       # ROJO (código de salida 1)
python -m src.drift.concept_drift                                 # gráfica docs/figuras/concept_drift.png
python -m src.drift.calibrar_umbrales                             # justifica los umbrales con lotes normales
```

Pruebas automatizadas:

```powershell
pytest -v                                                              # todo en verde
$env:LOTE="data/lotes/lote_derivado.csv"; pytest tests/test_data_drift.py -k lote   # ROJO
Remove-Item Env:LOTE
```

El monitoreo de concept drift también queda en MLflow, en el experimento **churn-monitoreo**: la métrica `f1_lote` por semana aparece como gráfica temporal.

### 6. Limpieza

```powershell
kubectl delete namespace churn
```
