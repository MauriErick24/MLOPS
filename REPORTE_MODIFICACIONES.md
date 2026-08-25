# Reporte de Modificaciones — MLOPS/deteccion_fraude

---

# Entrada 2026-08-25 — Capa de inferencia FastAPI

## Resumen

Se expuso el modelo promovido a Production por HTTP. El Model Registry
guardaba el modelo pero no el preprocesamiento que lo alimenta, así que un
modelo en Production no podía puntuar una transacción cruda.

## Archivos nuevos

| Archivo | Contenido |
|---|---|
| `deteccion_fraude/serving.py` | `ServingArtifacts` (scalers, `noise_mask`, features seleccionadas, umbrales), `ScoringModel` y `score_transactions()` |
| `deteccion_fraude/api/schemas.py` | Contratos Pydantic v2: `Transaction` (30 columnas), `PredictionRequest/Response` |
| `deteccion_fraude/api/model_loader.py` | Resolución de versión y flavor desde el Registry |
| `deteccion_fraude/api/app.py` | App FastAPI: `GET /`, `GET /health`, `POST /predict` |
| `tests/test_serving.py` | 11 tests de preprocesamiento y ventanas |
| `tests/test_api.py` | 14 tests de endpoints con modelo simulado |

## Archivos modificados

| Archivo | Cambio |
|---|---|
| `features.py` | `AmountStats`; `transform()` acepta `amount_stats` opcional |
| `dataset.py` | `PreparedData` gana el campo `amount_stats` |
| `modeling/pipeline.py` | `save_serving_artifacts()` |
| `modeling/train.py` | `fraude train` persiste artefactos; nuevo comando `fraude serve` |
| `requirements.txt` | `fastapi`, `uvicorn`, `pydantic`, `httpx` explícitos |
| `MLproject`, `Makefile` | Entry points y targets `promote` / `serve` |

## Bug corregido: `Amount_zscore` con lotes pequeños

**Archivo:** `deteccion_fraude/features.py`

**Problema:** `Amount_zscore` se calculaba con `scipy.stats.zscore` sobre el
lote. Con una sola transacción la desviación estándar es 0, el z-score sale
`NaN` y el `dropna()` del final de `transform()` descartaba la fila entera.
Una petición de una transacción — el caso de uso principal de la API —
devolvía cero resultados sin error visible.

**Fix:** `AmountStats` persiste `mean`, `std` y los bordes de bin del
entrenamiento. En inferencia se reutilizan en vez de recalcularlos por lote.
Los valores fuera del rango de entrenamiento se recortan con `np.clip` para
que `pd.cut` no produzca `NaN`.

**Sin impacto en entrenamiento:** durante `fit_transform` los estadísticos se
registran pero no se aplican; las matrices de entrenamiento son idénticas.

## Limitación conocida

`Amount_roll_*` y `Transaction_frequency` siguen calculándose sobre el lote
recibido, de modo que un lote pequeño no reproduce el contexto de
entrenamiento. Resolverlo requiere un feature store con el historial del
titular. Está documentado en `ARQUITECTURA.md` y `GUIA_EJECUCION.md`.

## Verificación

| Check | Estado |
|---|---|
| `pytest tests/` | 33/33 passed |
| `ruff check` | All checks passed |
| Entrenamiento completo | TabNet F1=0.7917 ROI=242%, LSTM F1=0.7692 ROI=192% |
| `fraude promote` | TabNet v1 promovido a Production |
| `POST /predict` con datos reales | 5/6 correctas (1 falso negativo con probabilidad 0.742 frente a umbral 0.792) |

---

# Entrada 2026-08-19 — Implementación OOP y corrección de bugs

## Resumen

Se implementó el 40% faltante del proyecto OOP (4 clases faltantes + 5 stubs CLI) y se corrigieron 2 bugs en la capa de datos del LSTM.

---

## Archivos modificados

### 1. `requirements.txt` — Dependencias unificadas

**Antes:** Solo herramientas básicas (loguru, pytest, ruff, tqdm, typer). Faltaban tensorflow, torch, pandas, numpy, scikit-learn, etc.

**Ahora:** Todas las dependencias del proyecto con versiones fijadas para compatibilidad con Python 3.10.

| Paquete | Versión | Motivo |
|---|---|---|
| imbalanced-learn | >=0.14 | Compatibilidad con scikit-learn 1.7 |
| numpy | >=1.24,<2.0 | Estabilidad con TensorFlow 2.18 |
| pandas | >=2.0,<3.0 | Compatibilidad |
| scikit-learn | >=1.3,<2.0 | Compatibilidad con imbalanced-learn |
| tensorflow | >=2.15,<2.19 | Versión estable para Python 3.10 |
| torch | >=2.0,<3.0 | Compatibilidad con pytorch-tabnet |

---

### 2. `deteccion_fraude/config.py` — Clase ExperimentConfig

**Antes:** Solo constantes de paths (PROJ_ROOT, DATA_DIR, etc.). La clase `ExperimentConfig` no existía.

**Ahora:** Clase `ExperimentConfig` con 11 atributos y 1 método:

| Atributo | Tipo | Valor | Usado por |
|---|---|---|---|
| `sequence_length` | int | 5 | lstm.py, pipeline.py |
| `smote_ratio` | float | 0.3 | lstm.py |
| `random_state` | int | 42 | lstm.py, feature_selection.py, pipeline.py |
| `false_negative_cost` | float | 150.0 | evaluation.py |
| `false_positive_cost` | float | 25.0 | evaluation.py |
| `min_lstm_features` | int | 30 | feature_selection.py |
| `models_dir` | Path | models/ | pipeline.py |
| `data_raw` | Path | data/raw/dataset.csv | — |
| `robust_columns` | ClassVar | 15 columnas | features.py |
| `standard_columns` | ClassVar | 7 columnas | features.py |
| `feature_columns` | ClassVar | 50 columnas | features.py, tests |
| `create_output_directories()` | method | — | pipeline.py |

---

### 3. `deteccion_fraude/dataset.py` — FraudDataset + PreparedData

**Antes:** Stub de Cookiecutter con `tqdm(range(10))`.

**Ahora:** 3 clases implementadas:

- **`DatasetSplits`** (dataclass): df_train, df_val, df_test
- **`FraudDataset`**: `load(path)` → read_csv + dropna + drop_duplicates; `split(df, config)` → train_test_split 70/15/15 estratificado
- **`PreparedData`** (dataclass): 15 atributos (X_train, X_validation, X_test, X_train_lstm, X_validation_lstm, X_test_lstm, X_balanced, y_train, y_validation, y_test, y_balanced, feature_names, class_weights, robust_scaler, standard_scaler) + método `apply_feature_selection()`

---

### 4. `deteccion_fraude/features.py` — FraudFeatureEngineer + FraudPreprocessor

**Antes:** Stub de Cookiecutter con `tqdm(range(10))`.

**Ahora:** 2 clases implementadas:

- **`FraudFeatureEngineer`**: `transform(df)` → crea 19 features derivadas (log, zscore, rolling, ratios, estadísticas de V) sin mutar input
- **`FraudPreprocessor`**: `fit_transform(splits)` → feature engineering + RobustScaler/StandardScaler (fit solo en train) + SMOTE + class_weights → PreparedData

---

### 5. `deteccion_fraude/modeling/train.py` — CLI entrenamiento

**Antes:** Stub de Cookiecutter con `tqdm(range(10))`.

**Ahora:** Conectado a `FraudDetectionPipeline`: carga datos → features → select_features → train → evaluate → save_artifacts.

---

### 6. `deteccion_fraude/modeling/predict.py` — CLI inferencia

**Antes:** Stub de Cookiecutter con `tqdm(range(10))`.

**Ahora:** Carga `results.pkl` y muestra métricas de cada modelo.

---

### 7. `deteccion_fraude/plots.py` — FraudVisualizer

**Antes:** Stub de Cookiecutter con `tqdm(range(10))`.

**Ahora:** Clase `FraudVisualizer` con:
- `plot_confusion_matrix(cm, model_name)` → heatmap de confusión
- `plot_results_summary(results)` → barras de métricas y ROI

---

### 8. `tests/test_data.py` — Tests de FraudDataset

**Antes:** `assert False` explícito.

**Ahora:** 4 tests:
- `test_load_drops_duplicates` → verifica que elimina duplicados
- `test_split_counts_sum_to_total` → verifica que train+val+test = total
- `test_split_preserves_class_proportion` → verifica stratificación
- `test_split_sorted_by_time` → verifica ordenamiento temporal

---

## Bugs corregidos

### Bug 1: selected_idx calculado contra array incorrecto

**Archivo:** `deteccion_fraude/dataset.py` — método `apply_feature_selection()`

**Problema:** `selected_idx` se calculaba contra `self.feature_names` (50 features originales) pero se usaba en `self.X_train` que ya había sido filtrado por `noise_mask` (menos columnas). Esto producía índices incorrectos cuando `noise_mask` eliminaba features.

**Antes:**
```python
all_names = list(self.feature_names)
selected_idx = [all_names.index(f) for f in selected_features]  # índices de 50
self.X_train = self.X_train[:, noise_mask]                       # X_train ahora tiene <50 cols
self.X_train_lstm = self.X_train[:, selected_idx]                # índices fuera de rango
```

**Ahora:**
```python
self.X_train = self.X_train[:, noise_mask]                       # primero filtrar
filtered_names = [n for n, keep in zip(self.feature_names, noise_mask) if keep]
selected_idx = [filtered_names.index(f) for f in selected_features]  # índices de filtered
self.X_train_lstm = self.X_train[:, selected_idx]                # índices correctos
```

---

### Bug 2: Doble compensación de desbalance en LSTM

**Archivo:** `deteccion_fraude/modeling/lstm.py` — método `fit()`

**Problema:** El LSTM aplicaba SMOTE (para balancear) Y `class_weight=data.class_weights` (peso ~300x para fraudes) simultáneamente. La combinación creaba un peso efectivo de ~2700x para la clase fraudulenta, causando que el modelo colapsara y predijera todo como negativo (F1=0.0).

**Cálculo del sobre-peso:**
- SMOTE: ~3x (sampling_strategy=0.3)
- FocalLoss alpha: 3x (alpha=0.75)
- class_weight: ~300x (ratio real 1:600)
- **Total: 3 × 3 × 300 = 2700x**

**Fix:** Eliminado `class_weight=data.class_weights` del `model.fit()`. SMOTE ya maneja el desbalance; class_weight es redundante.

---

## Verificación

| Check | Estado |
|---|---|
| `python -m pytest tests/` | 8/8 passed |
| `ruff check` | All checks passed |
| `ruff format --check` | 13 files already formatted |
| Pipeline ejecución | TabNet F1=0.72, LSTM requiere re-ejecución post-fix |
