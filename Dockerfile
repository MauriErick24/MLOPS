# syntax=docker/dockerfile:1

FROM python:3.10-slim

ARG TORCH_VERSION=2.5.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}+cpu" \
    && python -m pip install --requirement requirements.txt

ENV FRAUD_PROJECT_ROOT=/app

COPY pyproject.toml README.md LICENSE ./
COPY deteccion_fraude ./deteccion_fraude
COPY MLproject python_env.yaml ./
COPY tests ./tests

RUN find /app/deteccion_fraude /app/tests -type f -name '*.py' -exec chmod 0644 {} + \
    && python -m pip install --no-deps . \
    && groupadd --system fraud \
    && useradd --system --gid fraud --home-dir /app fraud \
    && mkdir -p /app/data/raw /app/models /app/reports/figures /mlflow/db /mlflow/artifacts \
    && chown -R fraud:fraud /app /mlflow

USER fraud

EXPOSE 8000

CMD ["fraude", "serve", "--host", "0.0.0.0", "--port", "8000"]
