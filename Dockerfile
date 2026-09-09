# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Metadatos del proyecto
LABEL maintainer="Data Science Team"
LABEL description="LogisticNetwork: Fulfillment Network Optimizer & What-If Simulator"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias de sistema (solver CBC de optimización y utilitarios)
RUN apt-get update && apt-get install -y --no-install-recommends \
    coinor-cbc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar archivos de configuración de dependencias
COPY pyproject.toml ./

# Instalar dependencias del proyecto
RUN pip install --upgrade pip && \
    pip install -e .

# Copiar código fuente, datos y modelos
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

# Exponer puerto predeterminado de Streamlit
EXPOSE 8501

# Healthcheck para monitorización de estado del contenedor
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando por defecto para iniciar el dashboard interactivo
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
