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
```

**Si tu venv tiene Python 3.14** (error con tensorflow), elimina y recrea:
```bash
rmdir /s /q venv
"C:\Users\Truji\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Estructura del proyecto

```
MLOPS/
├── data/raw/dataset.csv          ← Dataset original (creditcard.csv)
├── deteccion_fraude/
│   ├── config.py                 ← ExperimentConfig
│   ├── dataset.py                ← FraudDataset + DatasetSplits + PreparedData
│   ├── features.py               ← FraudFeatureEngineer + FraudPreprocessor
│   ├── evaluation.py             ← FraudModelEvaluator
│   ├── plots.py                  ← FraudVisualizer
│   └── modeling/
│       ├── pipeline.py           ← FraudDetectionPipeline (fachada)
│       ├── lstm.py               ← LSTMDetector + FocalLoss
│       ├── tabnet.py             ← TabNetDetector
│       ├── feature_selection.py  ← TabNetFeatureSelector
│       ├── train.py              ← CLI entrenamiento
│       └── predict.py            ← CLI inferencia
├── models/                       ← Modelos y artifacts generados
├── reports/figures/              ← Graficos generados
└── tests/                        ← Tests unitarios
```

---

## 3. Ejecucion del pipeline completo

### Opcion A: Script unificado (recomendado)

```bash
python deteccion_fraude/modeling/train.py
```

Ejecuta todo el pipeline en secuencia:
1. Carga y limpieza del dataset
2. Particion estratificada 70/15/15
3. Feature engineering (19 variables derivadas)
4. Escalado (RobustScaler + StandardScaler)
5. SMOTE (solo en train, ratio 0.3)
6. Seleccion de features con TabNet
7. Entrenamiento LSTM + TabNet
8. Evaluacion con ajuste de umbral por ROI
9. Guardado de modelos y artifacts en `models/`

### Opcion B: Paso a paso

```bash
# 1. Cargar y particionar datos
python deteccion_fraude/dataset.py

# 2. Generar features
python deteccion_fraude/features.py

# 3. Entrenar modelos
python deteccion_fraude/modeling/train.py

# 4. Ver resultados
python deteccion_fraude/modeling/predict.py

# 5. Generar graficos
python deteccion_fraude/plots.py
```

### Opcion C: Make (Linux/Mac)

```bash
make data          # Cargar dataset
make test          # Ejecutar tests
make lint          # Verificar lint
make format        # Formatear codigo
```

---

## 4. Ejecucion de tests

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

## 5. Lint y formato

```bash
# Verificar errores de lint
python -m ruff check

# Verificar formato
python -m ruff format --check

# Auto-corregir
python -m ruff check --fix
python -m ruff format
```

---

## 6. Configuracion del experimento

Todas las constantes estan en `deteccion_fraude/config.py`:

| Parametro | Valor | Descripcion |
|---|---|---|
| `random_state` | 42 | Semilla global |
| `sequence_length` | 5 | Ventanas LSTM |
| `smote_ratio` | 0.3 | Ratio minoritario tras SMOTE |
| `false_negative_cost` | 150 | Costo de fraude no detectado |
| `false_positive_cost` | 25 | Costo de falsa alarma |
| `min_lstm_features` | 30 | Minimo de features para LSTM |

---

## 7. Artefactos generados

Tras ejecutar el pipeline, se generan en `models/`:

| Archivo | Contenido |
|---|---|
| `lstm_fraud_detector.keras` | Modelo LSTM entrenado |
| `tabnet_fraud_detector/` | Modelo TabNet entrenado |
| `results.pkl` | Metricas, umbrales y feature importance |
| `scaler.pkl` | RobustScaler y StandardScaler fitted |

En `reports/figures/`:

| Archivo | Contenido |
|---|---|
| `cm_lstm.png` | Matriz de confusion LSTM |
| `cm_tabnet.png` | Matriz de confusion TabNet |
| `results_summary.png` | Comparativa de metricas y ROI |

---

## 8. Uso programatico

```python
from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import FraudDataset
from deteccion_fraude.features import FraudPreprocessor
from deteccion_fraude.modeling.pipeline import FraudDetectionPipeline

config = ExperimentConfig()
df = FraudDataset.load(config.data_raw)
splits = FraudDataset.split(df, config)

preprocessor = FraudPreprocessor(config)
data = preprocessor.fit_transform(splits)

pipeline = FraudDetectionPipeline(config, data)
pipeline.select_features().train().evaluate().save_artifacts()

print(pipeline.results)
```
