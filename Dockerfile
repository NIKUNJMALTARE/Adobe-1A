FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --default-timeout=100 --retries=5 --no-cache-dir -r requirements.txt

# Copy model first
COPY models /app/models

# Copy app
COPY main.py extract_utils.py /app/
COPY input /app/input
COPY output /app/output

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["python", "main.py"]
