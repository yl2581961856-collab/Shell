# syntax=docker/dockerfile:1

# NOTE: Replace BASE_IMAGE with your Ascend CANN/NPU runtime image.
# Current server userland is openEuler 24.03 (LTS).
# Example (for reference only): ascendai/cann:8.0.0-rc1-openeuler24.03
ARG BASE_IMAGE=openeuler/openeuler:24.03-lts
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app/asr

# If your base image already includes Python 3.11, you can remove this section.
# openEuler uses dnf; adjust for your base image as needed.
RUN dnf -y install \
    python3.11 python3.11-devel python3.11-pip \
    ca-certificates git \
    && dnf clean all

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# NPU single card note: index 1
# Adjust if your machine uses a different device index.
ENV ASCEND_DEVICE_ID=1 \
    NPU_VISIBLE_DEVICES=1

# Default bind (override at runtime)
ENV ASR_HOST=0.0.0.0 \
    ASR_PORT=6008

EXPOSE 6008

# Model/code paths are expected to be mounted at runtime:
# /app/data/models, /app/data/asr/SenseVoice
CMD ["bash", "shell/start_server.sh"]
