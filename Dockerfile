# Lightweight runtime image built on PyTorch CUDA base
FROM nvcr.io/nvidia/pytorch:25.05-py3

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    WHISPER_MODEL_PATH=/app/models/whisper_model

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-get update) && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    libatlas3-base \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Base Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# CosyVoice runtime dependencies (without copying model weights)
COPY CosyVoice/requirements.txt /tmp/cosyvoice-requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com -r /tmp/cosyvoice-requirements.txt \
    && rm /tmp/cosyvoice-requirements.txt

# Application source (selective copy to avoid bloating the image)
COPY src/ src/
COPY config/ config/
COPY tools/ tools/
COPY docs/ docs/
COPY CosyVoice/ CosyVoice/
COPY README.md README.md

# Create mount points for external models/embeddings
RUN mkdir -p /app/models/whisper_model /app/models/embeddings

EXPOSE 9090

CMD ["uvicorn", "src.mcp_server:create_app", "--factory", "--host", "0.0.0.0", "--port", "9090"]
