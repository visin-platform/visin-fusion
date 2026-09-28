# Training image: datasets and outputs are mounted at runtime.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch DATA_ROOT=/data VISIN_DATA_DIR=/data FUSION_OUTPUT_DIR=/outputs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
COPY . .
# The package version: .git is not in the build context, so image.yml passes it (the tag's version)
ARG VERSION=0.0.0
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install '.[train,visin]'
ENTRYPOINT ["visin-fusion", "run"]
CMD ["--help"]
