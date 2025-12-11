
"""
Model Evaluation Script for CV Threat Detection

Evaluates both YOLO models (yolov11l.pt and yolov8l.pt) on the test set using YOLO's built-in val() method.
Saves all metrics, plots, and annotated images in results/yolov11l_eval/ and results/yolov8l_eval/.
"""

import os
from ultralytics import YOLO
import yaml
import glob
import numpy as np
import shutil
import csv

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATA_YAML = os.path.join(PROJECT_DIR, "training_data", "data.yaml")

MODEL_PATHS = [
    os.path.join(MODELS_DIR, "yolov11l.pt"),
    os.path.join(MODELS_DIR, "yolov8l.pt")
]

def get_best_and_worst_images(pred_dir, metric='conf'):
    image_scores = []
    for img_path in glob.glob(os.path.join(pred_dir, '*.jpg')):
        image_scores.append((img_path, os.path.getsize(img_path)))
    if not image_scores:
        return [], []
    image_scores.sort(key=lambda x: x[1], reverse=True)
    best = [x[0] for x in image_scores[:3]]
    worst = [x[0] for x in image_scores[-3:]]
    return best, worst

def write_detailed_summary(metrics, out_dir, pred_dir, class_names):
    summary = []
    summary.append(f"# Model Evaluation Summary\n")
    summary.append(f"**Precision:** {metrics.get('metrics/precision(B)', 'N/A'):.3f}")
    summary.append(f"**Recall:** {metrics.get('metrics/recall(B)', 'N/A'):.3f}")
    summary.append(f"**mAP50:** {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
    summary.append(f"**mAP50-95:** {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    summary.append(f"**Fitness:** {metrics.get('fitness', 'N/A'):.3f}\n")
    summary.append(f"## Per-Class Results:")
    for i, cname in enumerate(class_names):
        p = metrics.get(f'metrics/precision-{i}', None)
        r = metrics.get(f'metrics/recall-{i}', None)
        m50 = metrics.get(f'metrics/mAP50-{i}', None)
        m95 = metrics.get(f'metrics/mAP50-95-{i}', None)
        if p is not None:
            summary.append(f"- **{cname}**: Precision={p:.3f}, Recall={r:.3f}, mAP50={m50:.3f}, mAP50-95={m95:.3f}")
    best, worst = get_best_and_worst_images(pred_dir)
    summary.append(f"\n## Best (most detections) images:")
    for img in best:
        summary.append(f"- {os.path.basename(img)}")
    summary.append(f"\n## Worst (least detections) images:")
    for img in worst:
        summary.append(f"- {os.path.basename(img)}")
    summary.append(f"\n## Suggestions:")
    summary.append("- Review worst images for missed detections or labeling errors.")
    summary.append("- If mAP50-95 is low, consider more data or augmentations.")
    summary.append("- Check confusion matrix and PR curves in this folder for more insight.")
    with open(os.path.join(out_dir, "DETAILED_SUMMARY.md"), "w") as f:
        f.write("\n".join(summary))

def write_eval_readme(out_dir, model_name):
    content = f"""# {model_name} Evaluation Results\n\nThis folder contains the full evaluation results for the {model_name} model, including:\n\n- Annotated images (predictions)\n- Metrics summary (YAML)\n- Plots (PR, confusion matrix, F1, etc.)\n- COCO-format and TXT-format results\n- DETAILED_SUMMARY.md (this file)\n- results.csv (per-image results)\n- args.yaml (evaluation arguments)\n- samples/ (selected output images)\n"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(content)

def get_class_names(data_yaml):
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])

def write_results_csv(pred_dir, out_dir):
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "num_boxes", "labels"])
        for img_path in glob.glob(os.path.join(pred_dir, '*.jpg')):
            label_path = img_path.replace('.jpg', '.txt')
            num_boxes = 0
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    num_boxes = len(lines)
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            labels.append(parts[0])
            writer.writerow([os.path.basename(img_path), num_boxes, ','.join(labels)])


def write_args_yaml(metrics, out_dir):
    args_path = os.path.join(out_dir, "args.yaml")
    # Try to get args from metrics or fallback to model if available
    args = getattr(metrics, 'args', None)
    if args is not None:
        with open(args_path, "w") as f:
            yaml.dump(args, f)
    else:
        # Write a note if not available
        with open(args_path, "w") as f:
            f.write("args not available from metrics object.\n")

def copy_yolo_outputs(yolo_dir, out_dir):
    # Move/copy all files from yolo_dir to out_dir (except for README.md, which we overwrite)
    for item in os.listdir(yolo_dir):
        s = os.path.join(yolo_dir, item)
        d = os.path.join(out_dir, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

def save_sample_images(pred_dir, out_dir, n=6):
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    imgs = glob.glob(os.path.join(pred_dir, '*.jpg'))
    imgs = imgs[:n] if len(imgs) >= n else imgs
    for img in imgs:
        shutil.copy2(img, os.path.join(sample_dir, os.path.basename(img)))


def evaluate_with_yolo(model_path, data_yaml, results_dir):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    out_dir = os.path.join(results_dir, f"{model_name}_eval")
    # Ensure unique output directory for each model (do not delete previous results)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print(f"\n[INFO] Evaluating {model_name}... Results will be saved to {out_dir}")
    model = YOLO(model_path)
    # Run YOLO's built-in validation (evaluation) method
    metrics = model.val(
        data=data_yaml,
        split="test",
        project=out_dir,
        name="",
        save_json=True,
        save_hybrid=True,
        save_txt=True,
        save_conf=True,
        save=True,
        plots=True
    )
    print(f"[INFO] {model_name} evaluation complete. Key metrics:")
    print(metrics.results_dict)
    # Copy all YOLO outputs into out_dir (if not already)
    yolo_dir = out_dir  # outputs are now in out_dir directly
    pred_dir = os.path.join(out_dir, "labels")
    # Save summary metrics, detailed summary, README, results.csv, args.yaml, and samples
    summary_path = os.path.join(out_dir, "summary_metrics.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(metrics.results_dict, f)
    class_names = get_class_names(data_yaml)
    write_detailed_summary(metrics.results_dict, out_dir, pred_dir, class_names)
    write_eval_readme(out_dir, model_name)
    write_results_csv(pred_dir, out_dir)
    write_args_yaml(metrics, out_dir)
    save_sample_images(pred_dir, out_dir, n=6)

if __name__ == "__main__":
    for model_path in MODEL_PATHS:
        evaluate_with_yolo(model_path, DATA_YAML, RESULTS_DIR)
