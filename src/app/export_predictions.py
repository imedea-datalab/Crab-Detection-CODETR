import os
import sys
import glob
import json
import uuid
import csv
import argparse
import traceback
import cv2
import torch
from tqdm import tqdm

# Add mmdetection to path so custom modules are loaded
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mmdet_path = os.path.join(project_root, 'mmdetection')
if mmdet_path not in sys.path:
    sys.path.insert(0, mmdet_path)

# PyTorch 2.6 Compatibility Patch
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from src.config.loader import load_config

def generate_csv_from_json(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    headers = [
        "image", "model_version", "prediction_id", "score", "label",
        "box_x", "box_y", "box_width", "box_height", "orig_width", "orig_height"
    ]

    rows = []
    for entry in data:
        if not entry or not isinstance(entry, dict):
            continue
            
        image_path = entry.get("data", {}).get("image", "")
        
        for prediction in entry.get("predictions", []):
            model_version = prediction.get("model_version", "")
            
            for result in prediction.get("result", []):
                val = result.get("value", {})
                labels_list = val.get("rectanglelabels", [])
                label = ", ".join(labels_list) if isinstance(labels_list, list) else ""
                
                row = {
                    "image": image_path,
                    "model_version": model_version,
                    "prediction_id": result.get("id", ""),
                    "score": result.get("score", ""),
                    "label": label,
                    "box_x": val.get("x", ""),
                    "box_y": val.get("y", ""),
                    "box_width": val.get("width", ""),
                    "box_height": val.get("height", ""),
                    "orig_width": result.get("original_width", ""),
                    "orig_height": result.get("original_height", "")
                }
                rows.append(row)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Successfully converted to CSV: {csv_path}")

def run_export(group_name):
    config = load_config()
    inf_config = config.get("inference", {})
    
    base_data_dir = "/app/data"
    dataset_dir = inf_config.get("dataset_dir", os.path.join(base_data_dir, "images"))
    
    image_dir = os.path.join(dataset_dir, group_name)
    json_dir = os.path.join(base_data_dir, "json", group_name)
    csv_dir = os.path.join(base_data_dir, "csv", group_name)
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    json_output_path = os.path.join(json_dir, "predictions.json")
    csv_output_path = os.path.join(csv_dir, "predictions.csv")
    
    checkpoint_path = inf_config.get("checkpoint_path", "/app/data/weights/best_coco_bbox_mAP_50_epoch_4.pth")
    conf_threshold = inf_config.get("conf_threshold", 0.05)
    model_config_path = os.path.join(project_root, "src", "config_models", "crab_co_dino_swin_l.py")
    
    print("="*50)
    print(f"Exporting Predictions for Group: {group_name}")
    print("="*50)
    print(f"Image Directory : {image_dir}")
    print(f"JSON Output     : {json_output_path}")
    print(f"CSV Output      : {csv_output_path}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Conf Thresh     : {conf_threshold}")
    print("="*50)

    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)

    # Grab all images
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
    image_paths.sort()
    
    if len(image_paths) == 0:
        print(f"No .jpg or .png images found in {image_dir}")
        sys.exit(1)

    print("Initializing model...")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    model = AutoDetectionModel.from_pretrained(
        model_type='mmdet',
        model_path=checkpoint_path,
        config_path=model_config_path,
        confidence_threshold=conf_threshold,
        device='cuda:0'
    )
    
    ls_tasks = []
    
    for img_path in tqdm(image_paths, desc="Running Inference"):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {filename}")
            continue

        H, W = img.shape[:2]

        try:
            result = get_sliced_prediction(
                img_path,
                model,
                slice_height=1000,
                slice_width=1000,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            
            valid_bboxes = []
            valid_scores = []
            for pred in result.object_prediction_list:
                valid_scores.append(pred.score.value)
                valid_bboxes.append([pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy])
                
        except Exception as e:
            print(f"Exception during prediction for {filename}:\n{traceback.format_exc()}")
            continue

        predictions = []
        for bbox, score in zip(valid_bboxes, valid_scores):
            x_min, y_min, x_max, y_max = bbox

            # Label Studio expects percentages
            x_perc = (x_min / W) * 100
            y_perc = (y_min / H) * 100
            w_perc = ((x_max - x_min) / W) * 100
            h_perc = ((y_max - y_min) / H) * 100

            predictions.append({
                "id": str(uuid.uuid4())[:10],
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": int(W),
                "original_height": int(H),
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": float(x_perc),
                    "y": float(y_perc),
                    "width": float(w_perc),
                    "height": float(h_perc),
                    "rectanglelabels": ["crab"]
                },
                "score": float(score)
            })

        # Format the relative path for Label Studio local storage
        # Path should be like: images/group1/img.jpg
        relative_path_posix = f"images/{group_name}/{filename}"

        ls_tasks.append({
            "data": {
                "image": f"/data/local-files/?d={relative_path_posix}"
            },
            "predictions": [{
                "model_version": f"CO-DETR-Swin-L",
                "result": predictions
            }]
        })

    # Save JSON
    with open(json_output_path, 'w') as f:
        json.dump(ls_tasks, f, indent=2)
    print(f"\nSaved Label Studio JSON to {json_output_path}")

    # Save CSV
    generate_csv_from_json(json_output_path, csv_output_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export CO-DETR predictions to Label Studio format.")
    parser.add_argument('--group-name', type=str, required=True, help="Name of the subfolder in data/images to process.")
    args = parser.parse_args()
    
    run_export(args.group_name)
