# Use NVIDIA PyTorch base image or standard Python
# MMDetection and OpenCV usually need system dependencies
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies required for OpenCV and compiling extensions
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

# Copy uv dependency files
COPY pyproject.toml ./

# Install dependencies using uv into a virtual environment at /app/.venv
# This creates a local .venv folder inside the container
RUN uv sync --no-dev

# Copy application source code and custom MMDetection
COPY src ./src
COPY mmdetection ./mmdetection
COPY config.default.yaml ./

# Command to run inference
CMD ["/app/.venv/bin/python", "-m", "src.app.inference"]
