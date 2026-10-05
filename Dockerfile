# --- Stage 1: Build Environment ---
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

# Install python and build tools
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

WORKDIR /app
COPY pyproject.toml ./

# Force MMCV to compile for all CUDA architectures even when no GPU is present during build.
# The +PTX flag allows the driver to JIT compile for newer GPUs (like Blackwell sm_120).
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV FORCE_CUDA="1"

# Create venv and install dependencies, forcing CUDA compilation for mmcv
RUN uv venv --python python3.10 && \
    uv pip install "setuptools<70" wheel numpy cython && \
    uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu128 && \
    uv sync --no-dev --no-build-isolation

# --- Stage 2: Final Runtime Image ---
FROM ubuntu:22.04

WORKDIR /app

# Install runtime OS dependencies and Python 3.10
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3.10 \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code and custom MMDetection
COPY src ./src
COPY mmdetection ./mmdetection
COPY config.default.yaml ./

# Command to run bulk visual inference
CMD ["/app/.venv/bin/python", "-m", "src.app.visualize_inference"]
