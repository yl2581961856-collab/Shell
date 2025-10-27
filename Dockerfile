FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    WHISPER_MODEL_PATH=/app/data/models/whisper_model

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/models/whisper_model /app/data/models/embeddings

EXPOSE 9090

CMD ["uvicorn", "src.mcp_server:create_app", "--factory", "--host", "0.0.0.0", "--port", "9090"]
