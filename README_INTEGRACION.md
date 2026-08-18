# Integración en el proyecto Cookiecutter Data Science

Copie el contenido de este paquete sobre la raíz de su proyecto `deteccion_fraude`.
La estructura resultante utiliza el paquete Python interno del mismo nombre:

```text
deteccion_fraude/
├── data/raw/creditcard.csv
├── deteccion_fraude/
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── evaluation.py
│   ├── plots.py
│   └── modeling/
├── notebooks/fraude_detection_final_OOP.ipynb
├── models/
└── tests/
```

Antes de abrir el notebook, instale el proyecto en modo editable desde la raíz:

```bash
python -m pip install -e .
python -m pip install -r requirements-oop.txt
```

Si el CSV está en otra ubicación, cambie solamente `data_file` al crear
`ExperimentConfig` en el notebook. No agregue rutas absolutas dentro de los módulos.

Los archivos `config.py`, `dataset.py`, `features.py` y `plots.py` son nombres
convencionales de Cookiecutter Data Science v2. Si ya contienen lógica propia,
integre las clases conservando sus constantes existentes en vez de reemplazar
el archivo sin revisar.
