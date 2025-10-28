FROM nvcr.io/nvidia/pytorch:25.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/bin:$PATH \
    http_proxy=http://172.16.26.131:7890 \
    https_proxy=http://172.16.26.131:7890 \
    HTTP_PROXY=http://172.16.26.131:7890 \
    HTTPS_PROXY=http://172.16.26.131:7890



RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get clean && apt-get update

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git




WORKDIR /app

COPY requirements.txt .
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src:${PYTHONPATH}

EXPOSE 9090

CMD ["uvicorn", "src.mcp_server:create_app", "--factory", "--host", "0.0.0.0", "--port", "9090"]