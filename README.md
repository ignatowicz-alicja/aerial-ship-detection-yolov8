# Aerial Ship Detection with YOLOv8

This repository implements a two-phase pipeline for detecting ships in aerial and satellite images using YOLOv8:

- **Phase 1** – a YOLOv8 object detector trained to locate regions likely to contain ships
- **Phase 2** – a YOLOv8 classifier trained to perform binary classification: whether a chip contains a ship or not.

This approach improves speed and accuracy by avoiding full-frame detection on massive satellite images.

---

## Project Structure

```text
aerial-ship-detection-yolov8/
│
├── training_yolov8_classifier/         # Phase 2 – classifier (ship vs non-ship)
│   ├── main.py                         # Training pipeline
│   ├── data_utils.py                   # Data loading and augmentation
│   ├── model_utils.py                  # Model and metrics
│   └── config.py                       # Paths and parameters
│
├── train_yolov8_detector/              # Phase 1 – YOLOv8 detector (bounding boxes)
│   ├── main.py                         # Training runner
│   ├── train.py                        # Training function
│   └── config.py                       # Detector configuration
│
├── Program for ship detection in full-scale satellite images.py  # Full-scene pipeline
├── SVM_experiment.py                   # Optional: classic ML classification (SVM)
└── README.md
```



## Installation

Python version: **3.10+**

Install required libraries:

pip install ultralytics==0.4.3 opencv-python matplotlib shapely tqdm pandas scikit-learn


## Used Datasets

This project uses two main datasets:

### 1. Ships/Vessels in Aerial Images  
A curated dataset of 26.9k satellite images annotated with YOLO-format bounding boxes for a single class: **ship**. It enables accurate and efficient training of object detectors for maritime vessel detection.
- Source: [Ships/Vessels in Aerial Images – Kaggle](https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images)


### 2. Ships in Satellite Imagery   
A labeled dataset of small 80×80 satellite chips, with binary labels: ship or no-ship.
- Source: [ShipsNet - Satellite Image Classification – Kaggle](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)
