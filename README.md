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

### 3.1. Build the Docker Image

First, build the Docker image (this will use `uv` internally to quickly resolve and install dependencies into a virtual environment). **This exact command is the same for Linux, Windows PowerShell, and Windows Command Prompt:**

> **Troubleshooting: Permission Denied**
> If you encounter an error like `ERROR: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock` while running the docker build command, you may need to add yourself to the docker group:
> ```bash
> sudo usermod -aG docker $USER
> newgrp docker
> ```

```bash
docker build -t crab-detection-codetr .
```
OR run it inthe background. .
```bash
# Create a directory ./data/logs/ first. 
mkdir -p ./data/logs/
nohup docker build -t crab-detection-codetr . > ./data/logs/docker_build_v2.log 2>&1 &
```


### 3.2. Run the Container (Evaluation Mode)

This mode runs the original evaluation script (`src/app/inference.py`). It compares model predictions against a ground-truth JSON (defined by `APP__inference__coco_json_path` in your `.env`) to calculate mAP, F1-score, precision, and recall.

**Where is the COCO JSON file?:** For CO-DETR (which requires COCO format JSON), the annotations are stored separately under the CO-DETR tiled dataset folder on thor, for example: `/datalocal/akshay/cangrejo/v1_2_4/codetr_tiling_multicrop/instances_valid.json`. Currently you dont have that becuse the adtaset which you have in akshay-desktop is yolo format.

Because we have a standardized `data/` folder in this repository, you only need to mount the single `data` directory to the container.

**Linux:**
```bash
docker run --rm \
    --gpus all \
    --user $(id -u):$(id -g) \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config.default.yaml:/app/config.default.yaml \
    crab-detection-codetr .venv/bin/python -m src.app.inference
```

**Windows (PowerShell):**
```powershell
docker run --rm `
    --gpus all `
    --env-file .env `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/config.default.yaml:/app/config.default.yaml" `
    crab-detection-codetr .venv/bin/python -m src.app.inference
```

**Windows (Command Prompt):**
```cmd
docker run --rm ^
    --gpus all ^
    --env-file .env ^
    -v "%cd%/data:/app/data" ^
    -v "%cd%/config.default.yaml:/app/config.default.yaml" ^
    crab-detection-codetr .venv/bin/python -m src.app.inference
```

### 3.3. Run the Container (Blind Inference & Export Mode)

This is the standard mode if you just want to run inference blindly on a folder of raw images (without a ground-truth JSON) and export the results to Label Studio JSON and CSV formats. You dont need  `APP__inference__coco_json_path` in your .env for this mode.

The outputs will be automatically saved in `data/json/<group_name>` and `data/csv/<group_name>`.

**Linux Command:**
```bash
docker run --rm \
    --gpus all \
    --user $(id -u):$(id -g) \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    crab-detection-codetr .venv/bin/python -m src.app.export_predictions --group-name "group1"
```

**Windows (PowerShell) Command:**
```powershell
docker run --rm `
    --gpus all `
    --env-file .env `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/config.default.yaml:/app/config.default.yaml" `
    crab-detection-codetr .venv/bin/python -m src.app.export_predictions --group-name "group1"
```

**Windows (Command Prompt) Command:**
```cmd
docker run --rm ^
    --gpus all ^
    --env-file .env ^
    -v "%cd%/data:/app/data" ^
    -v "%cd%/config.default.yaml:/app/config.default.yaml" ^
    crab-detection-codetr .venv/bin/python -m src.app.export_predictions --group-name "group1"
```

> **Note on `--group-name` vs `dataset_dir`:** 
> The `.env` file sets the root image folder (`APP__inference__dataset_dir=/app/data/images`). 
> However, to prevent mixing all your images into one massive JSON file (which makes labeling chaotic), the script expects you to process specific subfolders. By passing `--group-name "group1"`, the script will automatically combine the root path with the group name (e.g., `/app/data/images/group1`) and output the JSON directly into a matching `data/json/group1/` folder.

## 4. Visualizing with Label Studio

You can run Label Studio locally via Docker to view your raw images and overlay the generated JSON predictions. 
*(Note: If you are running Label Studio on a remote machine, you can use SSH tunneling to view it locally by running: `ssh -L 8080:localhost:8080 your_username@remote_host_ip`)*

### 4.1. Start the Label Studio Container

> **Troubleshooting: Permission Issues with Data Directory**
> The Label Studio container runs as a non-root user (UID `1001`) for security reasons, but it needs write access to your local `data` directory. If you get a permission error on boot, you need to change the group ownership to group 0 (which the container uses) and grant write permissions:
> ```bash
> sudo chown -R :0 data
> sudo chmod -R g+rwX data
> ```

**Linux:**
```bash
docker pull heartexlabs/label-studio:latest
docker run -it -p 8080:8080 -v "$(pwd)/data:/label-studio/data" -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data heartexlabs/label-studio:latest
```

**Windows (PowerShell):**
```powershell
docker pull heartexlabs/label-studio:latest
docker run -it -p 8080:8080 -v "${PWD}/data:/label-studio/data" -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data heartexlabs/label-studio:latest
```

**Windows (Command Prompt):**
```cmd
docker pull heartexlabs/label-studio:latest
docker run -it -p 8080:8080 -v "%cd%/data:/label-studio/data" -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data heartexlabs/label-studio:latest
```

### 4.2. Import Predictions into Label Studio
1. Open `http://localhost:8080` in your browser. Create a local account and a new project (e.g. "Crab Review").
2. Go to **Labeling Setup** > **Custom Template** (or Code) and paste this layout:
    ```xml
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="crab" background="red"/>
      </RectangleLabels>
    </View>
    ```
    Click **Save**.
3. **(Important) Whitelist the Images Directory:** Go to **Settings** > **Cloud Storage** > **Add Source Storage**.
    * **Storage Type:** Local files
    * **Absolute local path:** `/label-studio/data/images` (This allows Label Studio to serve the local images)
    * **Treat every bucket object as a source file:** Toggle this **OFF**.
    * Click **Save** (Do not click Sync).
4. **Import the JSON Predictions:** Click **Add Source Storage** again.
    * **Storage Type:** Local files
    * **Absolute local path:** `/label-studio/data/json/group1` (Change `group1` to your actual folder name)
    * **File Filter Regex:** `.*\.json$`
    * **Import Method:** **Tasks** (This extracts the pre-predicted bounding boxes from the JSON and overlays them onto the images).
    * **Treat every bucket object as a source file:** Toggle this **OFF**.
5. Click **Save & Sync**.

## 5. Development Notes

If you want to run this locally without Docker, ensure you create a virtual environment, install the dependencies using `uv sync`, and execute using the virtual environment python:

```bash
uv sync
.venv/bin/python -m src.app.inference
```
*(Warning: Running outside Docker may lead to MMDetection/CUDA compilation issues depending on your local host setup.)*

## 6. Troubleshooting Common Issues

### 6.1. CUDA or PyTorch Compatibility Errors
If you see an error like `NVIDIA ... is not compatible with the current PyTorch installation` or if PyTorch fails to initialize, ensure your host machine's NVIDIA driver is up to date. The Docker image uses PyTorch `cu128` (CUDA 12.8), which requires your host machine to have NVIDIA driver version **525.60.13 or newer**, regardless of how old your physical GPU is.

### 6.2. Out Of Memory (OOM) Errors on Inference
If your GPU still runs out of memory (OOM) during inference:
1. Ensure you are not running other heavy workloads on the GPU simultaneously.
2. The inference script uses SAHI to automatically slice large 4K images into `1000x1000` patches to prevent memory exhaustion. If you have a GPU with less than 8GB of VRAM and you still encounter an OOM error, you can edit `src/app/export_predictions.py` and reduce the `slice_height` and `slice_width` from `1000` to `512`.

### 6.3. Configuration Changes Have No Effect
If you change a value in `config.default.yaml` (such as `conf_threshold: 0.5`) but the model output doesn't change, ensure you are mounting the config file using the Docker volume flag:
`-v $(pwd)/config.default.yaml:/app/config.default.yaml`
Without this flag, Docker will run using the old config file that was "baked" into the image when you originally ran `docker build`.
