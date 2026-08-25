# Diagramas UML del Proyecto

Archivos Mermaid (.mmd) — visualizables en GitHub, GitLab, VS Code (extensión Markdown Preview Mermaid Support), o [mermaid.live](https://mermaid.live).

## Archivos

| Diagrama | Archivo | Descripción |
|---|---|---|
| Clases | `class_diagram.mmd` | Clases del dominio, serving y API; composiciones y dependencias |
| Secuencia (entrenamiento) | `sequence_diagram.mmd` | Flujo de `fraude train` hasta el registro en MLflow |
| Secuencia (inferencia) | `inference_sequence_diagram.mmd` | Arranque de la API y recorrido de un `POST /predict` |
| Componentes | `component_diagram.mmd` | Arquitectura por capas (CLI → Serving → Core → MLflow/DVC) |
| Flujo de Datos | `data_flow_diagram.mmd` | Del CSV crudo al veredicto servido por HTTP |

## Cómo visualizar

### Opción 1: VS Code
1. Instalar extensión **Markdown Preview Mermaid Support**
2. Abrir cualquier `.mmd` → Preview

### Opción 2: mermaid.live
1. Copiar el contenido del `.mmd`
2. Pegar en [mermaid.live](https://mermaid.live)

### Opción 3: GitHub/GitLab
Los diagramas Mermaid se renderizan automáticamente en archivos `.md`:
````markdown
```mermaid
<contenido del .mmd>
```
````
