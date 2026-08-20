# Diagramas UML del Proyecto

Archivos Mermaid (.mmd) — visualizables en GitHub, GitLab, VS Code (extensión Markdown Preview Mermaid Support), o [mermaid.live](https://mermaid.live).

## Archivos

| Diagrama | Archivo | Descripción |
|---|---|---|
| Clases | `class_diagram.mmd` | 13 clases, composiciones, dependencias |
| Secuencia | `sequence_diagram.mmd` | Flujo de ejecución de train.py |
| Componentes | `component_diagram.mmd` | Arquitectura por capas (CLI → Core → Modeling → DVC) |
| Flujo de Datos | `data_flow_diagram.mmd` | Transformaciones del dataset paso a paso |

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
