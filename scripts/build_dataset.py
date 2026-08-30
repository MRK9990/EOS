import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
import exifread
import cv2


# ============================================================
# EOS PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "processed_photos"
OUTPUT_CSV = PROJECT_ROOT / "data" / "eos_dataset.csv"

DATASET_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def rgb_to_lab_mean(rgb_roi):
    if rgb_roi is None or rgb_roi.size == 0:
        return None

    bgr = rgb_roi[..., ::-1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = float(np.mean(lab[..., 0]))
    a = float(np.mean(lab[..., 1]))
    b = float(np.mean(lab[..., 2]))
    return L, a, b


def extract_exif_data(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)

        iso = tags.get("EXIF ISOSpeedRatings", 0)
        exposure = tags.get("EXIF ExposureTime", 0)
        fnumber = tags.get("EXIF FNumber", 0)
        datetime_original = tags.get("EXIF DateTimeOriginal", None)

        # ISO
        if iso != 0:
            iso = int(str(iso))

        # Exposure
        if exposure != 0:
            num, den = str(exposure).split('/')
            exposure = float(num) / float(den)

        # FNumber
        if fnumber != 0:
            num, den = str(fnumber).split('/')
            fnumber = float(num) / float(den)

        hour = 0
        minute = 0

        if datetime_original:
            dt = str(datetime_original)
            time_part = dt.split(" ")[1]
            hour, minute, _ = map(int, time_part.split(":"))

        return iso, exposure, fnumber, hour, minute

    except Exception as e:
        print("EXIF read error:", e)
        return None


def parse_folder_tags(folder_name):
    """Parse environment tags from folder name format: code;wall_type;cloud_condition"""
    tag_white_walls = 0
    tag_clouds = 0
    tag_no_clouds = 0

    if not isinstance(folder_name, str):
        return tag_white_walls, tag_clouds, tag_no_clouds

    parts = folder_name.split(";")
    if len(parts) < 3:
        return tag_white_walls, tag_clouds, tag_no_clouds

    # [0] light code (ignore), [1] wall type, [2] cloud condition
    wall_type = parts[1].strip()
    cloud_type = parts[2].strip()

    if wall_type == "white_walls":
        tag_white_walls = 1

    if cloud_type == "clouds":
        tag_clouds = 1
    elif cloud_type == "no_clouds":
        tag_no_clouds = 1

    return tag_white_walls, tag_clouds, tag_no_clouds


rows = []
skipped = 0

for root, dirs, files in os.walk(DATASET_ROOT):
    if "meta.json" not in files:
        continue

    meta_path = os.path.join(root, "meta.json")
    img_path = os.path.join(root, "test_original.jpg")

    print(f"\nProcessing: {root}")

    if not os.path.isfile(img_path):
        print("  → No test_original.jpg")
        skipped += 1
        continue

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print("  → JSON error:", e)
        skipped += 1
        continue

    try:
        delta_list = meta["deltaE"]["per_roi"]["DE76"]

        if any(d is None for d in delta_list):
            print("  → Delta contains None")
            skipped += 1
            continue

        min_index = int(np.argmin(delta_list))
    except Exception as e:
        print("  → DeltaE error:", e)
        skipped += 1
        continue

    try:
        roi = meta["rois_test"][min_index]
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    except Exception as e:
        print("  → ROI error:", e)
        skipped += 1
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        if y+h > img_np.shape[0] or x+w > img_np.shape[1]:
            print("  → ROI out of bounds")
            skipped += 1
            continue

        roi_img = img_np[y:y+h, x:x+w]
        lab = rgb_to_lab_mean(roi_img)

        if lab is None:
            print("  → LAB None")
            skipped += 1
            continue

    except Exception as e:
        print("  → Image error:", e)
        skipped += 1
        continue

    exif_data = extract_exif_data(img_path)
    if exif_data is None:
        print("  → EXIF missing")
        skipped += 1
        continue

    L, a, b = lab
    iso, exposure, fnumber, hour, minute = exif_data

    # Lux
    lux_min = meta["lux_test"]["min"]
    lux_max = meta["lux_test"]["max"]
    lux_mean = (lux_min + lux_max) / 2

    # Lighting tags
    tags = meta["lighting_tags"]

    # Sliders (targets)
    sliders = meta["sliders"]

    # parse environmental tags from folder name
    folder_name = os.path.basename(root)
    tag_white_walls, tag_clouds, tag_no_clouds = parse_folder_tags(folder_name)

    row = {
        "L": L,
        "a": a,
        "b": b,
        "lux_min": lux_min,
        "lux_max": lux_max,
        "lux_mean": lux_mean,
        "ISO": iso,
        "exposure_time": exposure,
        "f_number": fnumber,
        "hour": hour,
        "minute": minute,
        "white_balance": sliders["white_balance"],
        "red_gain": sliders["red_gain"],
        "blue_gain": sliders["blue_gain"],
        "brightness": sliders["brightness"],
        "contrast": sliders["contrast"],
        "exposure_ev": sliders["exposure_ev"],
        "tag_white_walls": tag_white_walls,
        "tag_clouds": tag_clouds,
        "tag_no_clouds": tag_no_clouds
    }

    # one-hot для lighting_tags
    for tag in tags:
        row[f"tag_{tag.replace(' ', '_')}"] = 1

    rows.append(row)

df = pd.DataFrame(rows)
df.fillna(0, inplace=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Dataset created: {len(df)} samples")
print(f"Skipped: {skipped}")