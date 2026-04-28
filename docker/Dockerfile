# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvidia/cuda:12.6.0-devel-ubuntu22.04
FROM ${BASE_IMAGE} AS base

ARG BASE_IMAGE
ARG PYTHON_VERSION=3.10.20
ARG UV_EXTRA=cuda126

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:${PATH} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/root/.cache/huggingface \
    BASE_IMAGE=${BASE_IMAGE} \
    IMAGE_BASE_NAME=${BASE_IMAGE} \
    PYTHON_VERSION=${PYTHON_VERSION} \
    UV_EXTRA=${UV_EXTRA}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv python install ${PYTHON_VERSION}

FROM base AS deps

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra ${UV_EXTRA}

FROM base AS app

COPY --from=deps /opt/venv /opt/venv
COPY . /workspace
COPY docker/entrypoint.sh /usr/local/bin/taac-entrypoint

RUN chmod +x /usr/local/bin/taac-entrypoint
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra ${UV_EXTRA}

ENTRYPOINT ["taac-entrypoint"]
CMD ["/bin/bash"]
