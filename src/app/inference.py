import os
import sys
import json
import torch

# Add mmdetection to path so custom modules are loaded
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mmdet_path = os.path.join(project_root, 'mmdetection')
if mmdet_path not in sys.path:
    sys.path.insert(0, mmdet_path)

# PyTorch 2.6 Compatibility Patch
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

from mmdet.apis import init_detector, inference_detector
from torchvision.ops import box_iou
from tqdm import tqdm

from src.config.loader import load_config

def run_inference():
    config = load_config()
    
    # Load settings from config
    inf_config = config.get("inference", {})
    co_detr_config = config.get("v1_2_4", {}).get("co_detr", {})
    
    dataset_dir = inf_config.get("dataset_dir", "/app/data/images")
    coco_json_path = inf_config.get("coco_json_path", "/app/data/annotations/instances_valid.json")
    checkpoint_path = inf_config.get("checkpoint_path", "/app/data/weights/best_coco_bbox_mAP_50_epoch_4.pth")
    conf_threshold = inf_config.get("conf_threshold", 0.05)
    iou_threshold = inf_config.get("iou_threshold", 0.2)
    
    target_size = co_detr_config.get("target_size", 512)
    model_config_path = os.path.join(project_root, "src", "config_models", "crab_co_dino_swin_l.py")
    
    print("="*50)
    print("Crab Detection CO-DETR Inference")
    print("="*50)
    print(f"Dataset JSON : {coco_json_path}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Model Config : {model_config_path}")
    print(f"Conf Thresh  : {conf_threshold}")
    print(f"IoU Thresh   : {iou_threshold}")
    print(f"Target Size  : {target_size}")
    print("="*50)

    # 1. Load COCO ground truth
    if not os.path.exists(coco_json_path):
        print(f"Error: {coco_json_path} not found.")
        sys.exit(1)
        
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        
    images = {}
    for img in coco_data['images']:
        if target_size is not None and img['width'] != target_size:
            continue
        images[img['id']] = img
        
    gt_boxes = {img_id: [] for img_id in images.keys()}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in gt_boxes:
            x, y, w, h = ann['bbox']
            gt_boxes[img_id].append([x, y, x + w, y + h])

    # 2. Initialize the model
    print("Initializing model...")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    model = init_detector(model_config_path, checkpoint_path, device='cuda:0')
    
    tp, fp, fn = 0, 0, 0

    print(f"\nEvaluating on {len(images)} images...")
    
    # 3. Run evaluation
    for img_id, img_info in tqdm(images.items()):
        img_filename = os.path.basename(img_info['file_name'])
        # Try to resolve full path from dataset_dir
        img_path = os.path.join(dataset_dir, img_filename)
        
        if not os.path.exists(img_path):
            # Fallback to the path in json if it's absolute
            img_path = img_info['file_name']
            if not os.path.exists(img_path):
                print(f"Skipping {img_filename}, file not found.")
                continue

        try:
            result = inference_detector(model, img_path)
            pred_instances = result.pred_instances
            scores = pred_instances.scores
            valid_idx = scores >= conf_threshold
            pred_bboxes = pred_instances.bboxes[valid_idx].cpu()
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue
            
        gt = torch.tensor(gt_boxes[img_id])
        
        if len(gt) == 0 and len(pred_bboxes) == 0:
            continue
        elif len(gt) == 0:
            fp += len(pred_bboxes)
            continue
        elif len(pred_bboxes) == 0:
            fn += len(gt)
            continue
            
        ious = box_iou(pred_bboxes, gt)
        matched_gt = set()
        valid_scores = scores[valid_idx].cpu()
        sorted_indices = torch.argsort(valid_scores, descending=True)
        
        for pred_idx in sorted_indices:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(gt)):
                if gt_idx in matched_gt:
                    continue
                iou = ious[pred_idx, gt_idx].item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                tp += 1
            else:
                fp += 1
                
        missed_count = len(gt) - len(matched_gt)
        fn += missed_count
        
    print("\n" + "="*40)
    print("           EVALUATION RESULTS")
    print("="*40)
    
    if (tp + fp + fn) > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Total Crabs (GT)       : {tp + fn}")
        print(f"Total Detections       : {tp + fp}")
        print(f"True Positives (TP)    : {tp}")
        print(f"False Positives (FP)   : {fp}")
        print(f"False Negatives (FN)   : {fn}")
        print(f"Precision              : {precision:.4f}")
        print(f"Recall                 : {recall:.4f}")
        print(f"F1 Score               : {f1:.4f}")
    else:
        print("No evaluation data found.")
    print("="*40)

if __name__ == '__main__':
    run_inference()
