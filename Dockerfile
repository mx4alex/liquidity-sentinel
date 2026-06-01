FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY docs ./docs
COPY data/samples ./data/samples

RUN pip install --no-cache-dir -e ".[dev]"

ENV SENTINEL_DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["sentinel", "dashboard"]
