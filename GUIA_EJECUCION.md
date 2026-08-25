# Guia de Ejecucion — MLOPS/deteccion_fraude

## 1. Instalacion del entorno

> **Requisito:** Python **3.10** (no 3.14 — TensorFlow no soporta versiones mayores a 3.12)

```bash
# Clonar repositorio
git clone https://github.com/MauriErick24/MLOPS.git
cd MLOPS

# Crear entorno virtual con Python 3.10
python -m venv .venv

# Activar entorno
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
pip install -e .
```

**Si tu venv tiene Python 3.14** (error con tensorflow):
```bash
# Eliminar venv existente
rm -rf .venv                      # Linux/Mac
rmdir /s /q .venv                 # Windows

# Recrear con Python 3.10 explicito
python3.10 -m venv .venv          # Linux/Mac
py -3.10 -m venv .venv            # Windows (si py launcher esta instalado)
# O especificar la ruta completa:
# "C:\Users\TU_USUARIO\AppData\Local\Programs\Python\Python310\python.exe" -m venv .venv

source .venv/bin/activate         # Linux/Mac
.venv\Scripts\activate            # Windows

pip install -r requirements.txt
pip install -e .
```

---

## 2. Obtener datos (DVC)

El dataset esta trackeado con DVC. Despues de clonar:

```bash
# Instalar DVC (si no esta instalado)
pip install dvc dvc-gdrive

# Descargar datos desde Google Drive
dvc pull
```

Esto descarga `data/raw/creditcard.csv` (~150MB).

---

## 3. Estructura del proyecto

```
MLOPS/
├── MLproject                         ← Declaracion reproducible MLflow
├── python_env.yaml                   ← Entorno aislado
├── mlflow.db                         ← Backend store SQLite (se crea al primer run)
├── data/raw/creditcard.csv           ← Dataset original (via DVC)
├── deteccion_fraude/
│   ├── config.py                     ← ExperimentConfig
│   ├── dataset.py                    ← FraudDataset + DatasetSplits + PreparedData
│   ├── features.py                   ← FraudFeatureEngineer + FraudPreprocessor + AmountStats
│   ├── evaluation.py                 ← FraudModelEvaluator
│   ├── plots.py                      ← FraudVisualizer
│   ├── serving.py                    ← ServingArtifacts + ScoringModel (preprocesamiento en inferencia)
│   ├── tracking.py                   ← MLflowFraudTrainer + Wrappers + Lineage
│   ├── api/
│   │   ├── app.py                    ← FastAPI (/, /health, /predict)
│   │   ├── model_loader.py           ← Carga del modelo Production desde el Registry
│   │   └── schemas.py                ← Contratos Pydantic v2
│   └── modeling/
│       ├── pipeline.py               ← FraudDetectionPipeline (fachada)
│       ├── lstm.py                   ← LSTMDetector + FocalLoss
│       ├── tabnet.py                 ← TabNetDetector
│       ├── feature_selection.py      ← TabNetFeatureSelector
│       └── train.py                  ← CLI profesional (fraude train/promote/compare/serve)
├── models/                           ← Modelos y artifacts generados
├── reports/figures/                  ← Graficos generados
└── tests/                            ← Tests unitarios
```

---

## 4. CLI profesional

### Comandos disponibles

```bash
fraude --help                        # Ver comandos disponibles
fraude train --help                  # Ver opciones de entrenamiento
fraude promote --help                # Ver opciones de promocion
fraude compare --help                # Ver opciones de comparacion
fraude serve --help                  # Ver opciones de la API de inferencia
```

### Entrenar modelos

```bash
# Entrenar con configuracion por defecto
fraude train

# Entrenar con nombre de experimento personalizado
fraude train --experiment "v2"

# Entrenar con nombre de run personalizado
fraude train --run-name "experimento_enero"
```

### Promover modelo a Production

```bash
# Promover el ultimo run a Production
fraude promote

# Promover un run especifico
fraude promote --run-id abc123def456

# Promover por metrica diferente (default: f1)
fraude promote --metric roi
```

### Comparar runs

```bash
# Comparar todos los runs del experimento
fraude compare

# Comparar en un experimento especifico
fraude compare --experiment "v2"
```

### Alternativa: python -m

```bash
python -m deteccion_fraude.modeling.train train
python -m deteccion_fraude.modeling.train promote
python -m deteccion_fraude.modeling.train compare
python -m deteccion_fraude.modeling.train serve
```

---

## 5. MLflow Tracking

### Ver UI de MLflow

```bash
# Iniciar servidor MLflow (puerto 5000 por defecto)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# En otro terminal o con otro puerto
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

Abre `http://localhost:5000` en tu navegador.

### Que se registra automaticamente

| Tipo | Datos |
|------|-------|
| **Params** | sequence_length, smote_ratio, random_state, costs |
| **Metrics** | f1, precision, recall, roc_auc, roi, threshold, training_time |
| **Tags** | git_commit, dvc_status, environment, architecture, owner |
| **Models** | LSTM (Keras), TabNet (sklearn) con signature |
| **Artifacts** | 7 graficos (overview, feature importance, ROC/PR, confusion matrices) |

### Model Registry

Los modelos se promueven a "Production" con:

```bash
fraude promote
```

Esto:
1. Compara F1 de LSTM vs TabNet
2. Selecciona el ganador
3. Lo registra en MLflow Model Registry
4. Lo promueve a stage "Production"

### Cargar modelo desde Registry

```python
import mlflow

# Cargar modelo en Production
model = mlflow.pyfunc.load_model("models:/fraud_lstm/Production")

# Predecir
predictions = model.predict(X_new)
```

---

## 6. API de inferencia (FastAPI + Pydantic)

Sirve el modelo promovido a Production reaplicando el mismo preprocesamiento
que se uso en entrenamiento.

### Requisitos previos

```bash
fraude train      # entrena y genera models/serving_artifacts.pkl
fraude promote    # registra el ganador en Production
```

Sin estos dos pasos la API arranca en modo **Degradado** y `/predict` responde
503 con la instruccion correspondiente.

### Levantar el servidor

```bash
fraude serve                                  # http://127.0.0.1:8000
fraude serve --host 0.0.0.0 --port 8080       # exponer en la red
fraude serve --reload                         # desarrollo

# Alternativa directa con uvicorn
uvicorn deteccion_fraude.api.app:app --port 8000
```

Documentacion interactiva (Swagger UI) en `http://127.0.0.1:8000/docs`.

### Endpoints

| Metodo | Ruta | Descripcion |
|---|---|---|
| GET | `/` | Identidad del modelo servido (nombre, version, run_id, stage, flavor) |
| GET | `/health` | Readiness probe: 200 si esta Online, 503 si esta Degradado |
| POST | `/predict` | Puntua un lote cronologico de transacciones |

### Ejemplo de peticion

Cada transaccion lleva las 30 columnas del dataset: `Time`, `V1`..`V28`, `Amount`.
Los campos aceptan tanto el alias del dataset (`V14`) como el nombre snake_case (`v14`).

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"Time": 0.0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
       "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
       "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
       "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25, "V21": -0.02,
       "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13, "V26": -0.19,
       "V27": 0.13, "V28": -0.02, "Amount": 149.62}
    ]
  }'
```

### Respuesta

```json
{
  "model_metadata": {
    "name": "fraud_tabnet", "version": "3", "run_id": "abc123",
    "stage": "Production", "flavor": "tabnet"
  },
  "decision_threshold": 0.284,
  "total_predictions": 1,
  "scored_from_index": 0,
  "results": [
    {
      "index": 0,
      "prediction_code": 0,
      "diagnosis": "Legitima (0)",
      "fraud_probability": 0.0142,
      "confidence_score": 98.58,
      "high_risk_flag": 0
    }
  ],
  "message": "Inferencia completada con exito."
}
```

El `decision_threshold` no es 0.5: es el umbral que maximizo el ROI en
validacion durante el entrenamiento, persistido en los artefactos.

### Consideraciones de inferencia

| Tema | Comportamiento |
|---|---|
| **Orden del lote** | La lista se trata como una secuencia cronologica. |
| **Modelo LSTM** | Necesita `sequence_length` (5) transacciones para formar una ventana; las primeras 4 no reciben score y `scored_from_index` lo indica. Un lote menor a 5 devuelve 422. |
| **Modelo TabNet** | Puntua fila a fila, `scored_from_index` es 0. |
| **`Amount_zscore` / `Amount_log_cat`** | Usan los estadisticos fijados en entrenamiento (`AmountStats`), no los del lote, para que una sola transaccion sea puntuable. |
| **Variables de ventana** | `Amount_roll_*` y `Transaction_frequency` se calculan sobre el lote recibido: un lote pequeno no reproduce exactamente el contexto de entrenamiento. Eliminar esa diferencia requiere un feature store, fuera del alcance de esta capa. |

### Seleccion del modelo servido

Por defecto la API resuelve automaticamente cual de `fraud_tabnet` / `fraud_lstm`
esta en Production (si ambos, el promovido mas recientemente). Para forzar uno:

```bash
export FRAUD_MODEL_NAME=fraud_lstm     # Linux/Mac
$env:FRAUD_MODEL_NAME="fraud_lstm"     # Windows PowerShell
```

---

## 7. Ejecucion de tests

```bash
python -m pytest tests/ -v
```

Salida esperada:
```
tests/test_api.py ..............                                     [ 42%]
tests/test_data.py ....                                              [ 54%]
tests/test_evaluation.py ..                                          [ 60%]
tests/test_features.py ..                                            [ 66%]
tests/test_serving.py ...........                                    [100%]
33 passed
```

---

## 8. Lint y formato

```bash
# Verificar errores de lint
python -m ruff check

# Verificar formato
python -m ruff format --check

# Auto-corregir
python -m ruff check --fix
python -m ruff format

# Verificar tipos
python -m mypy deteccion_fraude/ --ignore-missing-imports
```

---

## 9. Configuracion del experimento

Todas las constantes estan en `deteccion_fraude/config.py`:

| Parametro | Valor | Descripcion |
|---|---|---|
| `random_state` | 42 | Semilla global |
| `sequence_length` | 5 | Ventanas LSTM |
| `smote_ratio` | 0.3 | Ratio minoritario tras SMOTE |
| `false_negative_cost` | 150 | Costo de fraude no detectado |
| `false_positive_cost` | 25 | Costo de falsa alarma |
| `min_lstm_features` | 30 | Minimo de features para LSTM |
| `mlflow_tracking_uri` | sqlite:///mlflow.db | Backend store MLflow |
| `mlflow_experiment_name` | deteccion_fraude | Nombre del experimento |

---

## 10. Artefactos generados

### En `models/` (locales)

| Archivo | Contenido |
|---|---|
| `lstm_fraud_detector.keras` | Modelo LSTM entrenado |
| `tabnet_fraud_detector/` | Modelo TabNet entrenado |
| `serving_artifacts.pkl` | Scalers ajustados, mascara de ruido, features seleccionadas, umbrales y `AmountStats`. Lo consume la API. |

### En MLflow (`mlflow.db`)

| Tipo | Contenido |
|---|---|
| **Params** | Hiperparametros del experimento |
| **Metrics** | F1, precision, recall, ROC-AUC, ROI |
| **Tags** | Git commit, DVC status, environment |
| **Models** | LSTM + TabNet con signature |
| **Figures** | 7 graficos de evaluacion |

### En `reports/figures/`

| Archivo | Contenido |
|---|---|
| `overview.png` | Distribucion de clases, montos, correlaciones |
| `feature_importance.png` | Top 25 features + importancia acumulada |
| `model_comparison.png` | Curvas ROC y Precision-Recall |
| `confusion_matrices.png` | Matrices de confusion LSTM y TabNet |
| `cm_lstm.png` | Matriz de confusion LSTM |
| `cm_tabnet.png` | Matriz de confusion TabNet |
| `results_summary.png` | Comparativa de metricas y ROI |

---

## 11. Uso programatico

```python
from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import FraudDataset
from deteccion_fraude.features import FraudPreprocessor
from deteccion_fraude.modeling.pipeline import FraudDetectionPipeline

config = ExperimentConfig()
df = FraudDataset.load(config.data_file)
splits = FraudDataset.split(df, config)

preprocessor = FraudPreprocessor(config)
data = preprocessor.fit_transform(splits)

pipeline = FraudDetectionPipeline(config, data)
pipeline.select_features().train().evaluate()

# Persistir el preprocesamiento que necesita la API
pipeline.save_serving_artifacts()

# Loggear en MLflow
run_id = pipeline.log_to_mlflow()

# Promover mejor modelo
winner = pipeline.trainer.promote_best_model(run_id)

print(f"Modelo ganador: {winner}")
print(f"MLflow run: {run_id}")
```

---

## 12. CI/CD (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: pip install -e .
      - run: ruff check deteccion_fraude/
      - run: mypy deteccion_fraude/ --ignore-missing-imports
      - run: pytest tests/ -v

  train:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: pip install -e .
      - run: fraude train --experiment "ci-${{ github.sha }}"
      - run: fraude promote
```

---

## 13. Solucion de problemas

### Error: TensorFlow no encuentra GPU
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Error: mlflow.db bloqueado
```bash
# Cerrar UI antes de entrenar, o usar otro puerto
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

### Error: dataset no encontrado
```bash
# Verificar que DVC esta configurado
dvc status

# Descargar datos
dvc pull
```

### Error: fraude command not found
```bash
# Reinstalar paquete en modo editable
pip install -e .
```

### Error: Model Registry vacio
```bash
# Primero entrenar, luego promover
fraude train
fraude promote
```

### La API responde 503 "No hay modelo en Production"
```bash
# El Registry no tiene ningun modelo promovido
fraude train
fraude promote
# Reiniciar la API: el modelo se carga una sola vez al arrancar
fraude serve
```

### La API responde 503 "Faltan los artefactos de inferencia"
```bash
# models/serving_artifacts.pkl se genera al final de 'fraude train'.
# Si entreno con una version anterior del pipeline, reentrene:
fraude train
```

### La API responde 422 con un lote de pocas transacciones
El modelo en Production es el LSTM, que necesita ventanas de 5 transacciones
consecutivas. Envie al menos 5 transacciones en `data`, o promueva TabNet:

```bash
$env:FRAUD_MODEL_NAME="fraud_tabnet"    # Windows PowerShell
fraude serve
```

### Error: Python 3.14 incompatible
```bash
# Eliminar venv y recrear con Python 3.10
rm -rf .venv                        # Linux/Mac
rmdir /s /q .venv                   # Windows

python3.10 -m venv .venv            # Linux/Mac
py -3.10 -m venv .venv              # Windows

source .venv/bin/activate           # Linux/Mac
.venv\Scripts\activate              # Windows

pip install -r requirements.txt
pip install -e .
```
