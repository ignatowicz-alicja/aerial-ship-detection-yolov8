import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
import os


DATA_PATH = r"\ścieżka\do\danych\pliku.json"
OUTPUT_DIR = r"folder\do\zapisywania\wyników"
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": 'L2-Hys'
}
PCA_COMPONENTS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42


def save_conf_matrix(y_true, y_pred, labels, name):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - SVM")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png"))
    plt.close()


with open(DATA_PATH, "r") as f:
    dataset = json.load(f)

X_raw = np.array(dataset["data"]).reshape(-1, 3, 80, 80).astype("uint8")
y = np.array(dataset["labels"])

print(f"Loaded {X_raw.shape[0]} samples")

#feature extraction 
def extract_hog_features(X):
    features = []
    for img in X:
        channels = [hog(img[i], **HOG_PARAMS) for i in range(3)]  # trzy kanały - R,G,B
        features.append(np.hstack(channels))
    return np.array(features)

print("Extracting HOG features...")
X_hog = extract_hog_features(X_raw)

#preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hog)

pca = PCA(n_components=PCA_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

#trening SVM
print("Training SVM...")
param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale", "auto"]}
svm = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring="accuracy", verbose=1)
svm.fit(X_train, y_train)

#evaluacja
y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

os.makedirs(OUTPUT_DIR, exist_ok=True)



report = classification_report(y_test, y_pred, target_names=["non-ship", "ship"], digits=4)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
print(report)


acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
save_conf_matrix(y_test, y_pred, labels=["non-ship", "ship"], name="svm_hog")

# Wykresy
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Liczba składowych głównych")
plt.ylabel("Skumulowana wyjaśniona wariancja")
plt.grid()
plt.title("Wariancja wyjaśniona przez PCA")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance.png"))
plt.close()
