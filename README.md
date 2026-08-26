# Integrantes
 - Vera Vargas Ariel Leandro 
 - Vilela Montoya María Fernanda
 - Molina Beltran Mauricio Erick
 - Roque Cerrogrande Edwin
 - Trujillo Montan Omar
 - Huanca Maldonado Rodrigo

# deteccion_fraude

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Deteccion de fraude en transacciones.

## Inicio rapido

Requiere **Python 3.10** (TensorFlow no soporta 3.13+).

```bash
py -3.10 -m venv .venv          # Windows
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

dvc pull                         # descarga data/raw/creditcard.csv

fraude train                     # entrena LSTM + TabNet, registra en MLflow
fraude promote                   # promueve el ganador a Production
fraude serve                     # API de inferencia en http://127.0.0.1:8000
```

Swagger UI en `http://127.0.0.1:8000/docs`.
Guia completa en [GUIA_EJECUCION.md](GUIA_EJECUCION.md), diseno en
[ARQUITECTURA.md](ARQUITECTURA.md).

## Docker Compose

Requiere Docker Desktop activo y `data/raw/creditcard.csv` descargado previamente
con `dvc pull`. La imagen es CPU y no incorpora datos, modelos ni secretos.

```bash
docker compose build
docker compose up -d mlflow

# Job manual: puede tardar considerablemente en CPU
docker compose run --rm trainer fraude train
docker compose run --rm trainer fraude promote

docker compose up -d api
docker compose ps
```

MLflow queda en `http://localhost:5000` y Swagger en
`http://localhost:8000/docs`. Despues de promover una nueva version:

```bash
docker compose restart api
```

Los runs, artifacts, modelos y reportes se conservan en volumenes. El dataset se
monta desde `./data` como solo lectura. Para elegir explicitamente un modelo se
puede definir `FRAUD_MODEL_NAME=fraud_tabnet` o `fraud_lstm`; el valor por defecto
es `auto`.

## Comandos

| Comando | Descripcion |
|---|---|
| `fraude train` | Entrena ambos modelos y registra el run en MLflow |
| `fraude compare` | Compara metricas entre runs del experimento |
| `fraude promote` | Promueve el mejor modelo a Production en el Registry |
| `fraude serve` | Levanta la API FastAPI sobre el modelo en Production |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | UI de MLflow |
| `pytest tests/ -v` | Suite de tests (34) |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make test`
├── MLproject          <- Declaracion reproducible de MLflow
├── python_env.yaml    <- Entorno aislado para MLflow Projects
├── README.md          <- The top-level README for developers using this project.
├── ARQUITECTURA.md    <- Diseno OOP, flujos de entrenamiento e inferencia
├── GUIA_EJECUCION.md  <- Guia de ejecucion paso a paso
│
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- creditcard.csv, versionado con DVC
│
├── docs               <- mkdocs project + diagramas UML (docs/uml/*.mmd)
│
├── models             <- Modelos entrenados y serving_artifacts.pkl
│
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         deteccion_fraude and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Graficos generados por FraudVisualizer
│
├── requirements.txt   <- Dependencias del proyecto
│
├── tests              <- Suite de tests (pytest)
│
└── deteccion_fraude   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes deteccion_fraude a Python module
    │
    ├── config.py               <- ExperimentConfig: rutas, semillas, costos, columnas
    │
    ├── dataset.py              <- FraudDataset, DatasetSplits, PreparedData
    │
    ├── features.py             <- FraudFeatureEngineer, FraudPreprocessor, AmountStats
    │
    ├── evaluation.py           <- FraudModelEvaluator: umbral por ROI y metricas
    │
    ├── plots.py                <- FraudVisualizer: EDA, curvas y matrices
    │
    ├── serving.py              <- ServingArtifacts, ScoringModel: preprocesamiento
    │                              persistido y scoring uniforme para inferencia
    │
    ├── tracking.py             <- MLflowFraudTrainer: tracking, registro y promocion
    │
    ├── api                     <- Capa de serving HTTP
    │   ├── app.py              <- FastAPI: GET /, GET /health, POST /predict
    │   ├── model_loader.py     <- Carga del modelo Production desde el Registry
    │   └── schemas.py          <- Contratos Pydantic v2
    │
    └── modeling
        ├── pipeline.py         <- FraudDetectionPipeline (fachada)
        ├── feature_selection.py<- TabNetFeatureSelector
        ├── lstm.py             <- LSTMDetector + focal loss
        ├── tabnet.py           <- TabNetDetector
        ├── predict.py          <- Resumen de metricas de los modelos guardados
        └── train.py            <- CLI: train / compare / promote / serve
```

--------

