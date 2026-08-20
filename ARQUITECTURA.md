# Arquitectura orientada a objetos

## Mapeo desde el notebook original

| Responsabilidad original | Clase nueva | Archivo |
|---|---|---|
| Rutas, semillas, costos y columnas | `ExperimentConfig` | `deteccion_fraude/config.py` |
| Lectura, limpieza y split | `FraudDataset` | `deteccion_fraude/dataset.py` |
| Variables derivadas | `FraudFeatureEngineer` | `deteccion_fraude/features.py` |
| Scalers, matrices y SMOTE | `FraudPreprocessor` | `deteccion_fraude/features.py` |
| Importancia de variables | `TabNetFeatureSelector` | `deteccion_fraude/modeling/feature_selection.py` |
| Arquitectura y entrenamiento LSTM | `LSTMDetector` | `deteccion_fraude/modeling/lstm.py` |
| Arquitectura y entrenamiento TabNet | `TabNetDetector` | `deteccion_fraude/modeling/tabnet.py` |
| Umbral, métricas y ROI | `FraudModelEvaluator` | `deteccion_fraude/evaluation.py` |
| EDA, curvas y matrices | `FraudVisualizer` | `deteccion_fraude/plots.py` |
| Coordinación del experimento | `FraudDetectionPipeline` | `deteccion_fraude/modeling/pipeline.py` |

## Flujo de objetos

1. `FraudDataset` entrega `DatasetSplits`.
2. `FraudPreprocessor` convierte los splits en `PreparedData`.
3. `FraudDetectionPipeline` recibe `PreparedData` por composición.
4. El selector modifica de forma sincronizada las matrices de `PreparedData`.
5. Los dos detectores entrenan y producen probabilidades.
6. El evaluador ajusta umbrales con validación y evalúa sobre prueba.
7. El pipeline guarda modelos, scalers y resultados en `models/`.

No se utiliza herencia porque los componentes no representan una relación
natural de tipo “es-un”. La composición expresa mejor la relación: un pipeline
**tiene** un selector, dos modelos y un evaluador.

## Decisiones metodológicas preservadas

- partición estratificada 70/15/15 con semilla 42;
- feature engineering independiente por split;
- ajuste de scalers exclusivamente en entrenamiento;
- SMOTE con `sampling_strategy=0.3` solo en entrenamiento;
- ventanas LSTM de longitud 5;
- hiperparámetros de LSTM y TabNet del notebook original;
- selección de umbral con validación;
- comparación final sobre el mismo subconjunto de prueba;
- costos FN=150 y FP=25 para el cálculo del ROI.
