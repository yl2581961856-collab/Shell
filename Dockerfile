# 使用 PyTorch 的 CUDA 版本
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 设置环境变量
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:${PYTHONPATH} \
    HF_HUB_OFFLINE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    libatlas3-base \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 拷贝并安装 Python 依赖
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# 安装 CosyVoice 的依赖（如果有）
COPY CosyVoice/requirements.txt /app/CosyVoice/requirements.txt
RUN pip install --no-cache-dir -r /app/CosyVoice/requirements.txt

# 如果需要安装 CosyVoice 的源代码依赖，可以选择 pip install -e
# RUN pip install -e /app/CosyVoice

# 下载并存储模型到本地（在构建镜像时下载）
# 假设你要下载 faster-whisper 的模型，并存储到 /app/models 目录
RUN mkdir -p /app/models/whisper_model && \
    python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cuda', download_root='/app/models/whisper_model')"

# 拷贝项目文件到容器
COPY . .

# 配置 faster-whisper 使用本地模型路径
ENV WHISPER_MODEL_PATH=/app/models/whisper_model

# 暴露端口
EXPOSE 9090

# 启动 FastAPI 服务
CMD ["uvicorn", "src.mcp_server:create_app", "--factory", "--host", "0.0.0.0", "--port", "9090"]
