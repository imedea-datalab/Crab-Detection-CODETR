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
*   **GPU Support:** To run inference quickly using your graphics card, you must pass `--gpus all` to Docker.
    - **Linux Users:** You *must* install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your host machine to bridge Docker with your physical GPU.
    ```bash
    # 1. Add the NVIDIA package repositories
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # 2. Update apt and install the toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # 3. Configure Docker to use the toolkit
    sudo nvidia-ctk runtime configure --runtime=docker

    # 4. Restart Docker so it recognizes the GPU
    sudo systemctl restart docker

    ```
    - **Windows Users:** If you are using Docker Desktop, you do **not** need to install any extra toolkit. Docker Desktop natively handles GPU pass-through automatically as long as you have the standard NVIDIA Windows drivers installed.
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
- `config.override.yaml`: **(Required)** The config loader expects a `config.override.yaml` file to exist, even if it's empty. Otherwise, it will crash. (An empty one has been created by default).

## 3. How to Run (Using Docker)

The most robust way to run this is using Docker, as it encapsulates all the complex `mmcv` and `torch` dependencies.

### 3.1. Build the Docker Image (Two Methods)

You can build the Docker image using either **Docker Compose** or standard **Docker CLI**. Both do the exact same thing under the hood:

| Method | Command | How It Works |
| :--- | :--- | :--- |
| **Method A: Docker Compose** *(Recommended)* | `docker compose build`<br>*(or `docker compose up --build`)* | Reads `docker-compose.yml`, finds `build: .`, runs the `Dockerfile`, and tags the image as `crab-detection-codetr:latest`. |
| **Method B: Classic Docker CLI** | `docker build -t crab-detection-codetr .` | Directly builds the `Dockerfile` in the current folder and tags it `crab-detection-codetr:latest`. |

#### How `docker compose up` vs `docker compose build` Works:
* **`docker compose build`**: **ONLY builds the image.** It does **not** start the container and does **not** run inference. Use this when you want to prepare the image ahead of time.
* **`docker compose up`**: **Runs inference immediately** using the already built image. It does **not** rebuild the image.
* **`docker compose up --build`**: **Builds/updates the image AND immediately runs inference** in one single command.

> **Troubleshooting: Permission Denied**
> If you encounter an error like `ERROR: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`, add yourself to the docker group:
> ```bash
> sudo usermod -aG docker $USER
> newgrp docker
> ```

```bash
# Method A: Build with Compose
docker compose build

# Method B: Build with Docker CLI
docker build -t crab-detection-codetr .
```


### 3.2. Run Bulk Visual & JSON Inference (Main Mode)

This is the standard mode to process a folder of raw images with Co-DETR (via SAHI). It runs inference and saves:
1. Annotated images with bounding boxes (`*_pred.jpg`).
2. Individual detection coordinate JSONs (`*_pred.json`).
3. A consolidated batch summary (`predictions.json`).

All outputs are saved automatically into a `results/` folder inside the input image directory.

#### Method A: Using Docker Compose (Easiest)
```bash
docker compose up
```

#### Method B: Using Docker CLI
**Linux:**
```bash
docker run --rm \
    --gpus all \
    --user $(id -u):$(id -g) \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config.default.yaml:/app/config.default.yaml:ro \
    crab-detection-codetr .venv/bin/python -m src.app.visualize_inference
```

**Windows (PowerShell):**
```powershell
docker run --rm `
    --gpus all `
    --env-file .env `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/config.default.yaml:/app/config.default.yaml:ro" `
    crab-detection-codetr .venv/bin/python -m src.app.visualize_inference
```

#### Mounting Custom Host Directories:
To run on any image folder on your computer without moving files into the project:
```bash
docker run --rm --gpus all \
    -v /path/to/my/images:/app/data/inference_samples \
    -v /path/to/my/checkpoint_dir:/app/data/model \
    crab-detection-codetr .venv/bin/python -m src.app.visualize_inference
```
All prediction images and JSONs will appear directly in `/path/to/my/images/results/`!

## 4. How to Distribute to Another Person

There are two primary ways to deliver this project to a collaborator or another machine:

### Option A: Pre-built Docker Image Tarball (`.tar.gz`) — (Recommended)
This is the cleanest and fastest delivery method. The recipient does **not** need Git, does **not** need PyTorch or CUDA installed on their host, does **not** need the `mmdetection` repository on their host, and does **not** need to wait for MMCV to compile. They only need Docker with NVIDIA GPU support.

1. **On your machine (Sender):**
   First, ensure the image is built with the latest code, then export it:
   ```bash
   # 1. Rebuild to ensure latest code is baked into the image
   docker compose build

   # 2. Export the image to a compressed tarball
   docker save crab-detection-codetr:latest | gzip > crab-detection-codetr.tar.gz
   ```
   Send them:
   - `crab-detection-codetr.tar.gz`
   - The model weights file: `best_coco_bbox_mAP_50_epoch_4.pth`

2. **On their machine (Recipient):**
   ```bash
   # 1. Load the Docker image
   docker load < crab-detection-codetr.tar.gz

   # 2. Run inference directly on any folder of images
   docker run --rm --gpus all \
       -v /path/to/their/images:/app/data/inference_samples \
       -v /path/to/weights_folder:/app/data/model \
       crab-detection-codetr:latest
   ```
   All predictions will appear automatically in `/path/to/their/images/results/`.

---

### Option B: Full Source Code Archive (`.zip`)
If the recipient wants the source code to modify it or run with `docker compose`:

1. **On your machine (Sender):**
   > **IMPORTANT:** Because `mmdetection/` is listed in `.gitignore`, a standard `git archive` will **NOT** include it! 
   > You must zip the directory directly (ensuring the `mmdetection/` folder is included):
   ```bash
   zip -r crab-detection-codetr.zip . \
       -x "data/dataset/*" \
       -x "data/model/*.pth" \
       -x ".venv/*" \
       -x "*/__pycache__/*"
   ```
   Send them the `.zip` along with the model weights file (`best_coco_bbox_mAP_50_epoch_4.pth`).

2. **On their machine (Recipient):**
   ```bash
   # 1. Unzip the archive
   unzip crab-detection-codetr.zip -d Crab-Detection-CODETR
   cd Crab-Detection-CODETR

   # 2. Place the weights file in data/model/
   mkdir -p data/model
   cp /path/to/best_coco_bbox_mAP_50_epoch_4.pth data/model/

   # 3. Place input images in data/inference_samples/
   mkdir -p data/inference_samples
   cp /path/to/my_images/*.jpg data/inference_samples/

   # 4. Build and run with Docker Compose in one shot:
   docker compose up --build
   ```
   *(Or build first with `docker compose build`, then run with `docker compose up` at their own convenience).*

## 5. Troubleshooting Common Issues

### 5.1. CUDA & Driver Compatibility
If you see an error like `NVIDIA ... is not compatible with the current PyTorch installation` or if PyTorch fails to initialize:
- Ensure your host machine's NVIDIA driver is up to date. The Docker container runs PyTorch with CUDA 12.8 (`cu128`), which requires a host NVIDIA driver of **550.x or 570.x+** (minimum 525.60.13).
- On newer **Blackwell GPUs** (e.g. RTX PRO 4000, `sm_120`), the host driver automatically JIT-compiles the embedded `+PTX` into native machine code.

### 5.2. Lower VRAM & Laptop GPUs (e.g., NVIDIA GTX 1060 6GB)
If deploying or running on a 6GB card like a **GTX 1060 Laptop GPU**:
1. **Driver Requirement**: Ensure the host has driver version **550 or 570+** installed. Pascal (`sm_61`) is supported by modern NVIDIA drivers and native kernels are pre-compiled into this image.
2. **Strict Slice Size Requirement**: You **MUST keep `slice_size: 512`** (set in `.env` as `APP__inference__slice_size=512` or passed via `--slice-size 512`).
   - At `slice_size: 512`, peak VRAM consumption is **~3.5 GB – 4.5 GB**, which fits comfortably inside the 6 GB VRAM of a GTX 1060.
   - Do **NOT** increase `slice_size` to 1000 on a 6 GB card, as the Swin-Large attention layers will trigger a PyTorch `OutOfMemoryError`.

### 5.3. Configuration Changes Have No Effect
If you change a value in `config.default.yaml` (such as `conf_threshold: 0.5`) but the model output doesn't change, ensure you are mounting the config file using the Docker volume flag:
`-v $(pwd)/config.default.yaml:/app/config.default.yaml`
Without this flag, Docker will run using the old config file that was "baked" into the image when you originally ran `docker build`.
