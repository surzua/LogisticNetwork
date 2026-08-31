# LogisticNetwork

Sistema modular de predicción de demanda y optimización de redes logísticas (MILP).

## Estructura del Proyecto

```text
LogisticNetwork/
├── data/
│   ├── 01_raw/          # Datos crudos sin modificar
│   ├── 02_intermediate/ # Datos limpios / preprocesados
│   └── 03_output/       # Predicciones generadas y resultados de optimización
├── notebooks/           # Exploración y prototipado rápido
├── src/                 # Código fuente modularizado
│   ├── data/            # Ingesta y feature engineering
│   ├── forecasting/     # Modelos predictivos
│   ├── optimization/    # Motor de optimización prescriptiva (MILP)
│   └── utils/           # Métricas, logging y utilidades
├── app/                 # Interfaz interactiva (Streamlit)
├── tests/               # Pruebas unitarias
├── Makefile             # Automatización de tareas
└── pyproject.toml       # Dependencias y configuración del entorno
```
