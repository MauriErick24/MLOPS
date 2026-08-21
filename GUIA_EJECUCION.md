# Guia de Ejecucion — MLOPS/deteccion_fraude

## 1. Instalacion del entorno

> **Requisito:** Python **3.10** (no 3.14 — TensorFlow no soporta versiones mayores a 3.12)

```bash
# Crear entorno virtual con Python 3.10
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Instalar paquete en modo editable (registra CLI fraude)
pip install -e .
```

**Si tu venv tiene Python 3.14** (error con tensorflow), elimina y recrea:
```bash
rmdir /s /q venv
"C:\Users\Truji\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 2. Estructura del proyecto

```
MLOPS/
├── MLproject                         ← Declaracion reproducible MLflow
├── python_env.yaml                   ← Entorno aislado
├── mlflow.db                         ← Backend store SQLite
├── data/raw/creditcard.csv           ← Dataset original
├── deteccion_fraude/
│   ├── config.py                     ← ExperimentConfig
│   ├── dataset.py                    ← FraudDataset + DatasetSplits + PreparedData
│   ├── features.py                   ← FraudFeatureEngineer + FraudPreprocessor
│   ├── evaluation.py                 ← FraudModelEvaluator
│   ├── plots.py                      ← FraudVisualizer
│   ├── tracking.py                   ← MLflowFraudTrainer + Wrappers + Lineage
│   └── modeling/
│       ├── pipeline.py               ← FraudDetectionPipeline (fachada)
│       ├── lstm.py                   ← LSTMDetector + FocalLoss
│       ├── tabnet.py                 ← TabNetDetector
│       ├── feature_selection.py      ← TabNetFeatureSelector
│       └── train.py                  ← CLI profesional (fraude train/promote/compare)
├── models/                           ← Modelos y artifacts generados
├── reports/figures/                  ← Graficos generados
└── tests/                            ← Tests unitarios
```

---

## 3. CLI profesional

### Comandos disponibles

```bash
fraude --help                        # Ver comandos disponibles
fraude train --help                  # Ver opciones de entrenamiento
fraude promote --help                # Ver opciones de promocion
fraude compare --help                # Ver opciones de comparacion
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
```

---

## 4. MLflow Tracking

### Ver UI de MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
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

## 5. Ejecucion de tests

```bash
python -m pytest tests/ -v
```

Salida esperada:
```
tests/test_data.py::test_load_drops_duplicates PASSED
tests/test_data.py::test_split_counts_sum_to_total PASSED
tests/test_data.py::test_split_preserves_class_proportion PASSED
tests/test_data.py::test_split_sorted_by_time PASSED
tests/test_evaluation.py::test_evaluator_returns_complete_confusion_matrix PASSED
tests/test_evaluation.py::test_threshold_is_inside_search_interval PASSED
tests/test_features.py::test_feature_engineer_creates_all_configured_columns PASSED
tests/test_features.py::test_feature_engineer_does_not_mutate_input PASSED
8 passed
```

---

## 6. Lint y formato

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

## 7. Configuracion del experimento

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

## 8. Artefactos generados

### En `models/` (locales)

| Archivo | Contenido |
|---|---|
| `lstm_fraud_detector.keras` | Modelo LSTM entrenado |
| `tabnet_fraud_detector/` | Modelo TabNet entrenado |

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

## 9. Uso programatico

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

# Loggear en MLflow
run_id = pipeline.log_to_mlflow()

# Promover mejor modelo
winner = pipeline.trainer.promote_best_model(run_id)

print(f"Modelo ganador: {winner}")
print(f"MLflow run: {run_id}")
```

---

## 10. CI/CD (GitHub Actions)

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

## 11. Solucion de problemas

### Error: TensorFlow no encuentra GPU
```bash
# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Error: mlflow.db bloqueado
```bash
# Cerrar UI antes de entrenar, o usar otro puerto
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

### Warning: Add type hints to predict method
```bash
# Ya corregido en tracking.py — FraudDecisionWrapper tiene type hints
# Si aparece, ejecutar:
pip install -e .
```

### Error: Model Registry vacio
```bash
# Primero entrenar, luego promover
fraude train
fraude promote
```
