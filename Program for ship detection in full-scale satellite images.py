import torch
from ultralytics import YOLO
import cv2
from PIL import Image
from pathlib import Path


SCENES_DIR = Path("data/scenes")  # folder ze zdjęciami stelitarnymi
DETECTION_MODEL = "models/detector.pt"  # model detekcji YOLOv8 (etap 1)
CLASSIFICATION_MODEL = "models/classifier.pt"  # model klasyfikacji YOLOv8-cls (etap 2)
CLASS_NAMES = ['no-ship', 'ship']
OUTPUT_DIR = SCENES_DIR.parent / "annotated"
OUTPUT_DIR.mkdir(exist_ok=True)

#parametry
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


detector = YOLO(DETECTION_MODEL)        
classifier = YOLO(CLASSIFICATION_MODEL) 


image_paths = list(SCENES_DIR.glob("*.png")) + list(SCENES_DIR.glob("*.jpg"))

for img_path in image_paths:
    print(f"[INFO] Processing: {img_path.name}")
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[WARN] Failed to load {img_path.name} – skipping.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #etap 1
    results = detector.predict(source=str(img_path), conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=DEVICE)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped = image_rgb[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        #etap 2
        pil_crop = Image.fromarray(cropped).resize((224, 224))  # przeskalowanie do wymaganego rozmiaru
        cls_result = classifier.predict(pil_crop, device=DEVICE, verbose=False)
        cls_id = int(cls_result[0].probs.top1)
        cls_label = CLASS_NAMES[cls_id]

        # pomijamy obiekty sklasyfikowane jako "no-ship"
        if cls_label != "ship":
            continue

        #wizualizacja detekcji
        color = (0, 255, 0)  # zielona ramka
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, cls_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    #zapis
    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), image)
    print(f"Saved: {out_path.name}")
