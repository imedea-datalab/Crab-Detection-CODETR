# Development History & Troubleshooting

This document tracks the significant technical challenges encountered during the containerization and deployment of the Crab-Detection-CODETR inference pipeline, and how they were resolved.

## 1. PyTorch & CUDA Architecture Compatibility (Blackwell GPUs)
**The Problem:**
When deploying the Docker container on a newer NVIDIA Blackwell GPU (RTX PRO 4000), PyTorch threw a warning: `NVIDIA RTX PRO 4000 Blackwell with CUDA capability sm_120 is not compatible with the current PyTorch installation`. When the script attempted to use MMDetection's custom CUDA extensions (like `ms_deform_attn_impl_forward`), it crashed because the kernels weren't compiled for `sm_120`.

**The Fix:**
- We upgraded the PyTorch backend from `cu124` to `cu128` (CUDA 12.8), which officially supports the `sm_120` architecture.
- We modified the Docker build process (Stage 1) to use `nvidia/cuda:12.8.0-devel-ubuntu22.04` and forced `mmcv` to pre-compile kernels for all major architectures by injecting `ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"`. This ensures the image is universally compatible across older (e.g. GTX 1060) and bleeding-edge (Blackwell) GPUs.

## 2. Missing OS Dependencies for OpenCV
**The Problem:**
During inference, OpenCV (`cv2`) failed to import with an `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`. The final runtime image was using a minimal `python-slim` base, which lacked basic graphics libraries.

**The Fix:**
- We added `libgl1` and `libglib2.0-0` to the `apt-get install` step in the final Docker runtime stage.

## 3. Broken Python Virtual Environment Symlinks (OS Mismatch)
**The Problem:**
After compiling dependencies in Stage 1 using an Ubuntu 22.04 builder, we copied the `.venv` directory to Stage 2, which was running a Debian-based `python:3.10-slim` image. When trying to run the container, we hit: `exec: ".venv/bin/python": stat .venv/bin/python: no such file or directory`. 
Because virtual environments use symlinks to the system Python binary (e.g. `/usr/bin/python3.10`), moving the `.venv` across different Linux distributions broke the symlinks (Debian stores Python in `/usr/local/bin/python`).

**The Fix:**
- We changed the Stage 2 base image to `ubuntu:22.04` (matching the builder) and manually installed `python3.10`. This ensured perfect symlink parity and OS library compatibility.

## 4. PyTorch Out Of Memory (OOM) on Large Images
**The Problem:**
When running inference on raw 4K drone images, the Swin-Large Vision Transformer inside CO-DETR crashed with a PyTorch `OutOfMemoryError`, requesting over 7.62 GiB for a single tensor allocation. Attention mechanisms scale quadratically, so feeding a full 4K image instantly exhausts even a 24GB VRAM GPU.

**The Fix:**
- We discovered the original project used **SAHI (Slicing Aided Hyper Inference)**.
- We rewrote `src/app/export_predictions.py` to replace standard MMDetection inference with SAHI's `get_sliced_prediction`, slicing the images into 1000x1000 patches before feeding them to the GPU, and automatically stitching the results back together.

## 5. SAHI Strict Pipeline Validation
**The Problem:**
After integrating SAHI, it threw a `ValueError: Resize is not found in the test pipeline`. SAHI strictly validates that MMDetection config files contain a `Resize` step, even though it feeds pre-sliced 1000x1000 patches. The original author used a messy runtime monkey-patch to inject this step into a temporary file.

**The Fix:**
- Instead of using a runtime hack, we cleanly updated `src/config_models/crab_co_dino_swin_l.py` to include `dict(type='Resize', scale=(1000, 1000), keep_ratio=True)` directly into the `test_pipeline`. Because the SAHI slices are already 1000x1000, this resize step acts as a harmless pass-through that satisfies SAHI's validation logic.

## 6. Dynamic Configuration Overrides (Avoiding "Baked-in" configs)
**The Problem:**
Modifying the host's `config.default.yaml` (e.g., changing `conf_threshold: 0.5`) had no effect on the container's output. The config file had been `COPY`'d into the Docker image during the build step, effectively freezing it in time.

**The Fix:**
- We updated the `docker run` commands in the README to mount the configuration file using a Docker volume: `-v $(pwd)/config.default.yaml:/app/config.default.yaml`. This allows users to tweak configuration parameters on the host and see the results immediately without rebuilding the image.
