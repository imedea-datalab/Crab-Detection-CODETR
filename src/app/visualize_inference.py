"""
Bulk visual inference and visualization module for Crab-Detection CO-DETR.

Performs sliced window inference (SAHI) with the Co-DETR (Co-DINO Swin-Large) model
on a directory of images and saves the annotated bounding-box visualizations into
a results directory.

Usage:
    # 1. As CLI command (reads defaults from config.default.yaml / .env):
    python -m src.app.visualize_inference

    # 2. With CLI argument overrides:
    python -m src.app.visualize_inference --input-dir /app/data/inference_samples --conf-threshold 0.5

    # 3. Inside Python scripts / Jupyter notebooks:
    from src.app.visualize_inference import visualize_crab, run_bulk_inference
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Any

# Headless matplotlib configuration to avoid GUI / DISPLAY errors in Docker & servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure mmdetection is in sys.path
MMDET_PATH = PROJECT_ROOT / 'mmdetection'
if str(MMDET_PATH) not in sys.path and MMDET_PATH.exists():
    sys.path.insert(0, str(MMDET_PATH))

# PyTorch 2.6 Compatibility Patch for MMDetection checkpoints
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from src.config.loader import load_config


def resolve_path(path_val: Optional[Any], default_val: Optional[Path] = None) -> Optional[Path]:
    """
    Intelligently resolves paths so they work both inside Docker (e.g. /app/data/...)
    and locally on the host machine.
    """
    if path_val is None or str(path_val).strip() == "":
        return default_val

    p = Path(str(path_val))

    # If the path exists as-is, return it
    if p.exists():
        return p

    # If running locally on host where /app doesn't exist, map /app/... to project_root / ...
    path_str = str(p)
    if path_str.startswith("/app/"):
        rel = path_str[len("/app/"):]
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate

    # Check relative to project root
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate

    return p


def load_sahi_model(
    checkpoint_path: str | Path,
    config_path: str | Path,
    conf_threshold: float = 0.5,
    device: Optional[str] = None
) -> AutoDetectionModel:
    """
    Loads and initializes the Co-DETR model using SAHI's AutoDetectionModel wrapper.
    """
    checkpoint_p = resolve_path(checkpoint_path)
    config_p = resolve_path(config_path)

    if checkpoint_p is None or not checkpoint_p.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. "
            f"Please verify your checkpoint path in config.default.yaml or .env (APP__inference__checkpoint_path)."
        )

    if config_p is None or not config_p.exists():
        raise FileNotFoundError(
            f"Model config file not found: {config_path}. "
            f"Please verify your model_config_path in config.default.yaml or .env."
        )

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Loading Co-DETR model on device [{device}]...")
    print(f"  Checkpoint : {checkpoint_p}")
    print(f"  Config     : {config_p}")

    model = AutoDetectionModel.from_pretrained(
        model_type='mmdet',
        model_path=str(checkpoint_p),
        config_path=str(config_p),
        confidence_threshold=conf_threshold,
        device=device
    )
    return model


def visualize_crab(
    img_path: str | Path,
    config_path: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    conf_threshold: float = 0.5,
    slice_size: int = 512,
    overlap: float = 0.2,
    device: Optional[str] = None,
    model: Optional[AutoDetectionModel] = None,
    show_label_name: bool = False,
) -> Tuple[plt.Figure, Any, AutoDetectionModel]:
    """
    Runs SAHI sliced inference on a single image and returns a matplotlib figure with bounding boxes.
    Maintains exact backward compatibility with notebook workflows.

    Args:
        img_path: Path to the input image file.
        config_path: Path to the MMDetection model config (defaults to project config if None).
        checkpoint_path: Path to the trained .pth checkpoint (defaults to config if None).
        conf_threshold: Confidence threshold for predictions.
        slice_size: Sliding window slice width/height for SAHI.
        overlap: Overlap ratio between sliding window slices.
        device: Device to run inference ('cuda:0' or 'cpu').
        model: Pre-loaded AutoDetectionModel instance (reuses model in GPU memory).
        show_label_name: If False, displays only the confidence score on bounding boxes.

    Returns:
        (fig, result, model): Matplotlib Figure, SAHI PredictionResult, and loaded Model.
    """
    img_p = Path(img_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    # 1. Initialize model if not already provided
    if model is None:
        cfg = load_config()
        inf_cfg = cfg.get("inference", {})

        if config_path is None:
            config_path = inf_cfg.get("model_config_path", "src/config_models/crab_co_dino_swin_l.py")
        if checkpoint_path is None:
            checkpoint_path = inf_cfg.get("checkpoint_path", "/app/data/model/best_coco_bbox_mAP_50_epoch_4.pth")

        model = load_sahi_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            conf_threshold=conf_threshold,
            device=device
        )

    # 2. Run sliced inference
    result = get_sliced_prediction(
        str(img_p),
        model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap
    )

    # 3. Export visuals to temporary file
    tmp_dir = "/tmp"
    if not os.path.exists(tmp_dir):
        tmp_dir = str(PROJECT_ROOT / "data" / "logs")
        os.makedirs(tmp_dir, exist_ok=True)

    if not show_label_name:
        for pred in result.object_prediction_list:
            object.__setattr__(pred.category, "name", "")

    temp_filename = f"sahi_temp_vis_{os.getpid()}"
    result.export_visuals(export_dir=tmp_dir, file_name=temp_filename, rect_th=3, text_size=2)

    vis_img_path = os.path.join(tmp_dir, f"{temp_filename}.png")
    vis_img = mpimg.imread(vis_img_path)

    # 4. Generate matplotlib plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(vis_img)
    ax.axis('off')

    num_detected = len(result.object_prediction_list)
    title_str = f"Crab Detections (Confidence > {conf_threshold}) - Preds: {num_detected}"
    ax.set_title(title_str, fontsize=14, pad=10)

    # Clean up temporary visual image
    try:
        if os.path.exists(vis_img_path):
            os.remove(vis_img_path)
    except OSError:
        pass

    return fig, result, model


def run_bulk_inference(
    input_dir: Optional[str | Path] = None,
    results_dir: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    conf_threshold: Optional[float] = None,
    slice_size: Optional[int] = None,
    overlap: Optional[float] = None,
    device: Optional[str] = None,
    dpi: int = 150
) -> Path:
    """
    Performs bulk inference over all images in input_dir and saves annotated images into results_dir.
    Pulls default values from config.default.yaml, config.override.yaml, and .env.
    """
    # Load configuration
    cfg = load_config()
    inf_cfg = cfg.get("inference", {})

    # Resolve settings with hierarchical fallbacks
    conf_threshold = conf_threshold if conf_threshold is not None else float(inf_cfg.get("conf_threshold", 0.5))
    slice_size = slice_size if slice_size is not None else int(inf_cfg.get("slice_size", 512))
    overlap = overlap if overlap is not None else float(inf_cfg.get("overlap_ratio", 0.2))
    dpi = int(inf_cfg.get("dpi", dpi))

    if device is None:
        device = inf_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_val = checkpoint_path or inf_cfg.get("checkpoint_path", "/app/data/model/best_coco_bbox_mAP_50_epoch_4.pth")
    config_val = config_path or inf_cfg.get("model_config_path", "src/config_models/crab_co_dino_swin_l.py")

    resolved_checkpoint = resolve_path(checkpoint_val)
    resolved_config = resolve_path(config_val)

    # Resolve input directory
    input_dir_val = input_dir or inf_cfg.get("input_dir") or inf_cfg.get("dataset_dir", "/app/data/inference_samples")
    resolved_input_dir = resolve_path(input_dir_val)

    # If the default input dir doesn't exist, check data/images/group1 as fallback
    if resolved_input_dir is None or not resolved_input_dir.exists():
        fallback_input = PROJECT_ROOT / "data" / "images" / "group1"
        if fallback_input.exists():
            resolved_input_dir = fallback_input
        else:
            raise FileNotFoundError(
                f"Input directory not found: {input_dir_val}. "
                f"Please ensure your images are located in that folder or specify --input-dir."
            )

    # Resolve results directory: defaults to <input_dir>/results
    results_dir_val = results_dir or inf_cfg.get("results_dir")
    if results_dir_val and str(results_dir_val).strip() != "":
        resolved_results_dir = resolve_path(results_dir_val, default_val=resolved_input_dir / "results")
    else:
        resolved_results_dir = resolved_input_dir / "results"

    resolved_results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      Crab Detection CO-DETR — Bulk Visual Inference")
    print("=" * 60)
    print(f"Input Directory  : {resolved_input_dir}")
    print(f"Results Directory: {resolved_results_dir}")
    print(f"Checkpoint       : {resolved_checkpoint}")
    print(f"Model Config     : {resolved_config}")
    print(f"Confidence Thresh: {conf_threshold}")
    print(f"Slice Window Size: {slice_size}x{slice_size} (overlap: {overlap})")
    print(f"Device           : {device}")
    print(f"Output DPI       : {dpi}")
    print("=" * 60)

    # Find image files (case-insensitive extensions)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted([
        f for f in resolved_input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])

    if not image_files:
        print(f"No image files ({', '.join(valid_extensions)}) found in {resolved_input_dir}")
        return resolved_results_dir

    print(f"\nFound {len(image_files)} image(s) to process.")

    # Load model once, keep in GPU VRAM
    model = load_sahi_model(
        checkpoint_path=resolved_checkpoint,
        config_path=resolved_config,
        conf_threshold=conf_threshold,
        device=device
    )

    total_crabs = 0
    all_results_data = []

    print("\nRunning inference and generating annotated visualizations...")
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images"), 1):
        fig, result, model = visualize_crab(
            img_path=img_path,
            config_path=resolved_config,
            checkpoint_path=resolved_checkpoint,
            conf_threshold=conf_threshold,
            slice_size=slice_size,
            overlap=overlap,
            device=device,
            model=model
        )

        # 1. Save annotated image
        save_path = resolved_results_dir / f"{img_path.stem}_pred.jpg"
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

        # 2. Extract bounding box coordinates, scores, and details for JSON export
        detections = []
        for pred in result.object_prediction_list:
            bbox = [float(pred.bbox.minx), float(pred.bbox.miny), float(pred.bbox.maxx), float(pred.bbox.maxy)]
            score = float(pred.score.value)
            detections.append({
                "score": round(score, 4),
                "bbox": [round(c, 2) for c in bbox],
                "width": round(bbox[2] - bbox[0], 2),
                "height": round(bbox[3] - bbox[1], 2),
                "category": "crab"
            })

        img_data = {
            "image": img_path.name,
            "crab_count": len(detections),
            "detections": detections
        }
        all_results_data.append(img_data)

        # Save individual per-image JSON file alongside the image
        json_save_path = resolved_results_dir / f"{img_path.stem}_pred.json"
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(img_data, f, indent=2)

        num_detected = len(detections)
        total_crabs += num_detected
        tqdm.write(f"[{idx}/{len(image_files)}] {img_path.name} -> {save_path.name} & {json_save_path.name} (Found {num_detected} crabs)")

    # 3. Save consolidated JSON summary for the entire batch
    summary_json_path = resolved_results_dir / "predictions.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": len(image_files),
            "total_crabs": total_crabs,
            "confidence_threshold": conf_threshold,
            "slice_size": slice_size,
            "overlap_ratio": overlap,
            "results": all_results_data
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("                      INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total Images Processed : {len(image_files)}")
    print(f"Total Crabs Detected   : {total_crabs}")
    print(f"Average Crabs / Image  : {total_crabs / len(image_files):.2f}")
    print(f"Saved Results Location : {resolved_results_dir}")
    print(f"Summary JSON File      : {summary_json_path}")
    print("=" * 60)

    return resolved_results_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run bulk CO-DETR inference on an image folder and save visualized bounding boxes."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default=None,
        help="Path to folder containing images (defaults to APP__inference__input_dir or config.default.yaml)"
    )
    parser.add_argument(
        "-o", "--results-dir",
        type=str,
        default=None,
        help="Path to save prediction images (defaults to <input-dir>/results)"
    )
    parser.add_argument(
        "-c", "--checkpoint-path",
        type=str,
        default=None,
        help="Path to trained model .pth checkpoint"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to MMDetection model config file"
    )
    parser.add_argument(
        "-t", "--conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold for bounding box detection (e.g. 0.5)"
    )
    parser.add_argument(
        "-s", "--slice-size",
        type=int,
        default=None,
        help="SAHI sliding window slice size in pixels (e.g. 512)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="SAHI slice overlap ratio (e.g. 0.2)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        help="Inference compute device: 'cuda:0' or 'cpu'"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved output images (default: 150)"
    )

    args = parser.parse_args()

    run_bulk_inference(
        input_dir=args.input_dir,
        results_dir=args.results_dir,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        conf_threshold=args.conf_threshold,
        slice_size=args.slice_size,
        overlap=args.overlap,
        device=args.device,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
