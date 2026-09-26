# Training image: code and dependencies only. Datasets are mounted at /data, run outputs at
# /outputs, and downloaded backbone weights are cached at /cache (see docker-compose.yml).
#
#   docker build -t visin-fusion .                                  # GPU (CUDA 13.0 wheels)
#   docker build -t visin-fusion:cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
#
# The CUDA 13.0 torch wheels run on the A100 (sm_80) and on Blackwell cards (sm_120), and
# bring their own CUDA libraries: the host only needs the NVIDIA driver and container toolkit.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    DATA_ROOT=/data \
    VISIN_DATA_DIR=/data \
    FUSION_OUTPUT_DIR=/outputs

# OpenCV's runtime libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first, from the index for this build (GPU or CPU); it is the largest layer and changes least
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY . .

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
