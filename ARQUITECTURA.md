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

## Capas añadidas sobre el notebook

Estas responsabilidades no existían en el notebook original: nacen de llevar el
experimento a un ciclo MLOps completo.

| Responsabilidad | Clase | Archivo |
|---|---|---|
| Tracking, registro y promoción en MLflow | `MLflowFraudTrainer` | `deteccion_fraude/tracking.py` |
| Estadísticos de `Amount` fijados en entrenamiento | `AmountStats` | `deteccion_fraude/features.py` |
| Preprocesamiento persistido para inferencia | `ServingArtifacts` | `deteccion_fraude/serving.py` |
| Scoring uniforme entre flavors de MLflow | `ScoringModel` | `deteccion_fraude/serving.py` |
| Contratos HTTP de entrada y salida | `Transaction`, `PredictionRequest`, `PredictionResponse` | `deteccion_fraude/api/schemas.py` |
| Carga del modelo en Production | `load_model`, `get_model_metadata` | `deteccion_fraude/api/model_loader.py` |
| Exposición HTTP | app FastAPI | `deteccion_fraude/api/app.py` |

## Flujo de objetos (entrenamiento)

1. `FraudDataset` entrega `DatasetSplits`.
2. `FraudPreprocessor` convierte los splits en `PreparedData`.
3. `FraudDetectionPipeline` recibe `PreparedData` por composición.
4. El selector modifica de forma sincronizada las matrices de `PreparedData`.
5. Los dos detectores entrenan y producen probabilidades.
6. El evaluador ajusta umbrales con validación y evalúa sobre prueba.
7. El pipeline guarda los modelos entrenados en `models/`.
8. `save_serving_artifacts()` persiste `models/serving_artifacts.pkl`.
9. `MLflowFraudTrainer` registra params, métricas, tags, modelos y figuras.

No se utiliza herencia porque los componentes no representan una relación
natural de tipo “es-un”. La composición expresa mejor la relación: un pipeline
**tiene** un selector, dos modelos y un evaluador.

## Flujo de objetos (inferencia)

El Model Registry guarda el modelo pero **no** el preprocesamiento que lo
alimenta. `ServingArtifacts` cierra esa brecha: sin scalers ajustados, máscara
de ruido y umbrales, un modelo en Production no puede puntuar una transacción
cruda.

1. Al arrancar, la app carga una vez el modelo (`load_model`) y los artefactos
   (`load_artifacts`). Si falta alguno, arranca **Degradada** y responde 503.
2. `POST /predict` valida el lote con Pydantic y lo vuelca a un `DataFrame`
   con los nombres de columna del dataset (`by_alias=True`).
3. `ServingArtifacts.build_matrix()` reaplica feature engineering, escalado y
   máscara de ruido, en el mismo orden que el entrenamiento.
4. Si el modelo es LSTM, `lstm_windows()` arma ventanas deslizantes de
   `sequence_length`; si es TabNet, la matriz va fila a fila.
5. `ScoringModel.score()` devuelve probabilidades sin importar el flavor.
6. El umbral aplicado es el que maximizó el ROI en validación, no 0.5.

### Separación entre dominio y transporte

`serving.py` no importa nada de FastAPI y `api/` no contiene lógica de
preprocesamiento. Esa frontera permite testear el scoring sin levantar un
servidor y reutilizarlo desde un batch job o un consumidor de cola.

## Limitación conocida del serving

`Amount_zscore` y `Amount_log_cat` dependían del lote: con una sola transacción
el z-score era `NaN` y la fila se descartaba. `AmountStats` fija esos
estadísticos en entrenamiento y los reaplica en inferencia.

Queda una diferencia sin resolver: `Amount_roll_*` y `Transaction_frequency`
siguen calculándose sobre el lote recibido, de modo que un lote pequeño no
reproduce el contexto de entrenamiento. Eliminarla exige un feature store con
el historial del titular, fuera del alcance de esta capa.

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

`AmountStats` no altera ninguna de estas decisiones: durante `fit_transform`
los estadísticos se **registran**, no se aplican. El entrenamiento produce
exactamente las mismas matrices que antes; los valores sólo se consumen en
inferencia.
