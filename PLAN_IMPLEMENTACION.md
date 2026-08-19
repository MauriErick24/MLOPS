# Plan de Implementación — MLOPS/deteccion_fraude

Corrección e implementación del 40% faltante del proyecto OOP.

---

## Fase 0: Unificar dependencias
- [x] Agregar `tensorflow, torch, pytorch-tabnet, imbalanced-learn, scikit-learn, pandas, numpy, matplotlib, seaborn, scipy` a `requirements.txt`
- [x] Verificar que el Makefile instala correctamente con `make requirements`

---

## Fase 1: `ExperimentConfig` en `config.py`
- [x] Crear clase `ExperimentConfig` con 8 atributos:
  - `sequence_length = 5`
  - `smote_ratio = 0.3`
  - `random_state = 42`
  - `false_negative_cost = 150`
  - `false_positive_cost = 25`
  - `min_lstm_features = 30`
  - `models_dir = PROJ_ROOT / "models"`
  - `feature_columns` (50 columnas: 28 PCA + 15 robust + 7 standard)
  - `robust_columns` (15 columnas)
  - `standard_columns` (7 columnas)
  - `data_raw = RAW_DATA_DIR / "dataset.csv"`
- [x] Implementar método `create_output_directories()`

---

## Fase 2: `FraudDataset` + `DatasetSplits` + `PreparedData` en `dataset.py`
- [x] Crear dataclass `DatasetSplits(df_train, df_val, df_test)`
- [x] Crear clase `FraudDataset`:
  - `load(path)` → read_csv + dropna + drop_duplicates
  - `split(df, config)` → train_test_split 70/15/15 estratificado, ordenado por Time
- [x] Crear dataclass `PreparedData` con 15 atributos:
  - Matrices: `X_train, X_validation, X_test` (n, 50)
  - Matrices LSTM: `X_train_lstm, X_validation_lstm, X_test_lstm` (n, 50)
  - SMOTE: `X_balanced, y_balanced`
  - Labels: `y_train, y_validation, y_test`
  - Metadata: `feature_names, class_weights, robust_scaler, standard_scaler`
  - Método: `apply_feature_selection(noise_mask, selected_features)`

---

## Fase 3: `FraudFeatureEngineer` + `FraudPreprocessor` en `features.py`
- [x] Crear `FraudFeatureEngineer(config)`:
  - `transform(df)` → crea 19 features derivadas sin mutar input
- [x] Crear `FraudPreprocessor(config)`:
  - `fit_transform(splits)` → feature engineering + scaling + SMOTE + class weights + PreparedData

---

## Fase 4: Tests corregidos
- [x] `test_data.py`: reemplazar `assert False` con tests de FraudDataset
- [x] Verificar que `test_features.py` pasa con FraudFeatureEngineer implementada
- [x] Verificar que `test_evaluation.py` pasa con ExperimentConfig implementada

---

## Fase 5: CLI stubs conectados a OOP
- [x] `dataset.py`: conectar a FraudDataset.load() + FraudDataset.split()
- [x] `features.py`: conectar a FraudFeatureEngineer + FraudPreprocessor
- [x] `modeling/train.py`: conectar a FraudDetectionPipeline
- [x] `modeling/predict.py`: conectar a inferencia con modelos guardados
- [x] `plots.py`: implementar FraudVisualizer

---

## Fase 6: Verificación final
- [x] `pip install -r requirements.txt` sin errores
- [x] `python -m pytest tests/` — 8/8 tests pasan
- [x] `ruff check` — All checks passed!
- [x] `ruff format --check` — 13 files already formatted

---

## Fase 7: Bugs corregidos
- [x] **Bug 1:** `selected_idx` calculado contra array de 50 features pero usado en array filtrado (< 50). Corregido en `dataset.py:apply_feature_selection()` — ahora calcula índices después del filtrado por noise_mask.
- [x] **Bug 2:** Doble compensación de desbalance en LSTM — SMOTE + class_weight (300x) + FocalLoss alpha (3x) = peso efectivo ~2700x, causando colapso del modelo (F1=0). Corregido en `lstm.py:fit()` — eliminado `class_weight=data.class_weights`.
