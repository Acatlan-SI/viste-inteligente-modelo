FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código y los modelos (.joblib y .onnx)
COPY . .

ENV MODEL_VERSION=v1
ENV MODELS_PATH=models/modelo.joblib
ENV GCS_BUCKET=viste-inteligente-bucket
ENV PORT=8080

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]