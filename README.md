# Crab-Detection-CODETR

## 1. Description

Crab-Detection-CODETR is a self-contained, Dockerized inference engine for the v1.2.4 CO-DETR (Swin-L) crab detection model. It provides an isolated environment to run object detection inference on large image sequences to replicate the baseline F1 metrics (e.g. `0.7317`).

The project uses a custom `mmdetection` library bundled inside the repository and executes in a Python virtual environment built via `uv`.

### 1.1. Folder structure (Quick)
- `src/app/`: Main inference code (`inference.py`).
- `src/config/`: Configuration loader.
- `src/config_models/`: Contains the specific CO-DETR PyTorch configuration file (`crab_co_dino_swin_l.py`).
- `mmdetection/`: The locally copied MMDetection framework library.
- `Dockerfile`: Defines the image for running inference cleanly.

## 2. Before Running

### 2.1. Prerequisites
*   **Docker:** A platform for running the inference container.
    - Recommended: Docker Engine 20.10+ with Docker Compose plugin.
*   **MMDetection Library:** The CO-DETR model requires the MMDetection framework. Since it is huge (2000+ files), it is ignored by Git. **Before building the Docker image**, you must either:
    - Copy the `mmdetection/` folder from the original `Crab-Detection` repository into the root of this project.
    - OR run: `git clone https://github.com/open-mmlab/mmdetection.git` in the root of this project.
*   **Dataset & Weights:** Because the image dataset and model weights (`best_coco_bbox_mAP_50_epoch_4.pth`) are massive, they are **not** baked into the Docker image. You must have them available on your local machine to mount into the container at runtime.

### 2.2. Environment Variables
This project uses `.env` files to override path configurations at runtime. By default, the inference script expects the dataset and weights to be mounted inside `/app/data` in the container.

Copy the example file to get started:
```bash
cp .env.example .env
```
Inside `.env`, you can see the expected paths:
```env
APP__inference__dataset_dir=/app/data/images
APP__inference__coco_json_path=/app/data/annotations/instances_valid.json
APP__inference__checkpoint_path=/app/data/weights/best_coco_bbox_mAP_50_epoch_4.pth
```

### 2.3. Configuration Files
- `config.default.yaml`: Contains the non-sensitive evaluation settings (Target size, Confidence thresholds, IoU thresholds, and backbone definition).

## 3. How to Run (Using Docker)

The most robust way to run this is using Docker, as it encapsulates all the complex `mmcv` and `torch` dependencies.

### 3.1. Build the Docker Image

First, build the Docker image (this will use `uv` internally to quickly resolve and install dependencies into a virtual environment):

```bash
docker build -t crab-detection-codetr .
```

### 3.2. Run the Container

Run the Docker container, mounting your local data directories into the expected `/app/data` paths inside the container. 

For example, if your weights and images are located locally at `/datalocal/akshay/`, you would run:

```bash
docker run --rm \
    --gpus all \
    --env-file .env \
    -v /home/atiwari/Downloads/projects/Crab-Detection/data/results/modelsOrCheckpoints/v1_2_4/20260630_003855:/app/data/weights \
    -v /datalocal/akshay/cangrejo/v1_2_4/codetr_tiling_multicrop:/app/data/annotations \
    -v /datalocal/akshay/cangrejo/v0_1/data_newsplit_full_json/valid/images:/app/data/images \
    crab-detection-codetr
```

**What this does:**
1. Maps your local model checkpoint to `/app/data/weights`.
2. Maps your local dataset JSON folder to `/app/data/annotations`.
3. Maps your local image folder to `/app/data/images`.
4. Runs `src/app/inference.py` directly using the container's `.venv/bin/python`.

## 4. Development Notes

If you want to run this locally without Docker, ensure you create a virtual environment, install the dependencies using `uv sync`, and execute using the virtual environment python:

```bash
uv sync
.venv/bin/python -m src.app.inference
```
*(Warning: Running outside Docker may lead to MMDetection/CUDA compilation issues depending on your local host setup.)*
