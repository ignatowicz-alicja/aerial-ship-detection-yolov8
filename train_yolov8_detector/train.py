import datetime
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from config import RUNS_DIR, IMG_SIZE, DATA_YAML, DATA_ROOT

def ensure_data_yaml():
    with open(DATA_YAML, "w") as f:
        f.write(f"path: {DATA_ROOT}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("names: ['ship']\n")
    return DATA_YAML

def train_detector():
    yaml_path = ensure_data_yaml()
    timestamp = datetime.datetime.now().strftime("f2_%Y-%m-%d_%H-%M-%S")
    model = YOLO("yolov8m.pt")
    results = model.train(
        data=yaml_path,
        imgsz=IMG_SIZE,
        epochs=50,
        batch=16,
        project=RUNS_DIR,
        name=timestamp,
        verbose=True,
        optimizer="AdamW",
        lr0=0.003,
    )

    run_dir = Path(RUNS_DIR) / timestamp
    metrics = results.results_dict
    print("\n Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    pd.DataFrame([metrics]).to_csv(run_dir / "results.csv", index=False)

    plt.figure(figsize=(10, 6))
    for metric in ["train/box_loss", "train/cls_loss", "metrics/mAP50", "metrics/precision", "metrics/recall"]:
        if metric in results.metrics:
            plt.plot(results.metrics[metric], label=metric)
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(run_dir / "metrics_plot.png")
    plt.close()

    print(f"[✓] Training completed → {run_dir}")
