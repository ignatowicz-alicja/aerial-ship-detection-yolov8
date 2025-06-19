import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
    precision_recall_curve
)
from ultralytics import YOLO
from pathlib import Path

def train_model(data_root: Path, run_dir: Path):
    model = YOLO("yolov8n-cls.pt")
    model.train(data=str(data_root / "images"), epochs=100, imgsz=80, batch=64,
                project=str(run_dir.parent), name=run_dir.name, verbose=False)
    return model

def evaluate(model: YOLO, data_root: Path, run_dir: Path):
    test_ship = list((data_root / "images" / "test" / "ship").glob("*.png"))
    test_bg   = list((data_root / "images" / "test" / "background").glob("*.png"))

    X, y_true = [], []
    for p in test_ship + test_bg:
        X.append(cv2.imread(str(p)))
        y_true.append(1 if "ship" in p.parts else 0)

    y_prob = []
    for img in X:
        res = model(img, verbose=False)[0]
        vec = res.probs.data.squeeze().cpu().numpy() if hasattr(res.probs, "data") else res.probs.numpy()
        y_prob.append(float(vec[1]))

    y_pred = [int(p >= 0.5) for p in y_prob]

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["bg", "ship"]).plot(cmap="Blues")
    plt.title("Confusion matrix"); plt.savefig(run_dir / "confusion_matrix.png"); plt.close()

    prec, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, prec); plt.title("Precision-Recall"); plt.savefig(run_dir / "pr_curve.png"); plt.close()

    report = classification_report(y_true, y_pred, target_names=["background", "ship"], output_dict=True)
    pd.DataFrame(report).transpose().to_json(run_dir / "classification_report.json", indent=2)
