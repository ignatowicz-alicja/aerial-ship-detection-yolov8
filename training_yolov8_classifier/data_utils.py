import os, json, random, shutil
from pathlib import Path
from typing import List
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import CHIPS_PER_IMAGE, VAL_RATIO, TEST_RATIO

def _mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_shipsnet(json_path: str, out_dir: Path):
    if list(out_dir.glob("ship/*.png")):
        print("[INFO] ShipsNet already extracted – skipping")
        return

    print("[INFO] Extracting ShipsNet chips …")

    if json_path.lower().endswith('.csv'):
        df = pd.read_csv(json_path)
        images = df.iloc[:, :-1].values.tolist()
        labels = df.iloc[:, -1].tolist()
    else:
        data = json.load(open(json_path))
        images = data.get("data") or data.get("images")
        labels = data.get("label") or data.get("labels")
        if images is None or labels is None:
            raise KeyError("ShipsNet JSON lacks expected keys.")

    ship_dir = out_dir / "ship"; _mkdir(ship_dir)
    bg_dir   = out_dir / "background"; _mkdir(bg_dir)

    for idx, (flat, lab) in tqdm(enumerate(zip(images, labels)), desc="ShipsNet", unit="img"):
        img = np.asarray(flat, dtype=np.uint8).reshape(3, 80, 80).transpose(1, 2, 0)
        cls_dir = ship_dir if lab == 1 else bg_dir
        cv2.imwrite(str(cls_dir / f"sn_{idx:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def generate_bg_chips(aerial_root: str, out_dir: Path):
    if list(out_dir.glob("*.png")):
        print("[INFO] Background chips already exist – skipping")
        return

    print("[INFO] Generating background chips …")
    _mkdir(out_dir)

    def iou(box, boxes):
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        for bx in boxes:
            xx1, yy1, xx2, yy2 = bx
            inter = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
            if area and inter / area > 0.05:
                return True
        return False

    img_dirs = [Path(aerial_root) / split / "images" for split in ("train", "test")]
    for img_dir in img_dirs:
        for img_path in tqdm(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")), desc=img_dir.parent.name):
            label_path = Path(str(img_path).replace("images", "labels").rsplit(".", 1)[0] + ".txt")
            if not label_path.exists():
                continue
            img = cv2.imread(str(img_path)); H, W = img.shape[:2]
            bboxes = []
            for line in open(label_path):
                _, xc, yc, w, h = map(float, line.split())
                bboxes.append([(xc - w/2)*W, (yc - h/2)*H, (xc + w/2)*W, (yc + h/2)*H])
            count = tries = 0
            while count < CHIPS_PER_IMAGE and tries < CHIPS_PER_IMAGE*10:
                tries += 1
                x = random.randint(0, W-80); y = random.randint(0, H-80)
                if not iou([x, y, x+80, y+80], bboxes):
                    cv2.imwrite(str(out_dir / f"bg_{img_path.stem}_{count}.png"), img[y:y+80, x:x+80])
                    count += 1

def merge_and_split(dst_root: Path, shipsnet_dir: Path, bg_dir: Path):
    ship_src = shipsnet_dir / "ship"
    bg_src   = list((shipsnet_dir / "background").glob("*.png")) + list(bg_dir.glob("*.png"))
    ships = list(ship_src.glob("*.png")); random.shuffle(ships); random.shuffle(bg_src)

    def _copy(lst: List[Path], dest: Path):
        _mkdir(dest)
        for p in lst:
            shutil.copy(p, dest / p.name)

    def split(arr):
        n = len(arr); val_n = int(n*VAL_RATIO); test_n = int(n*TEST_RATIO)
        return arr[test_n+val_n:], arr[:val_n], arr[val_n:val_n+test_n]

    s_train, s_val, s_test = split(ships)
    b_train, b_val, b_test = split(bg_src)

    for split_name, sh, bg in [("train", s_train, s_train + b_train), ("val", s_val, b_val), ("test", s_test, b_test)]:
        _copy(sh, dst_root / "images" / split_name / "ship")
        _copy(bg, dst_root / "images" / split_name / "background")
