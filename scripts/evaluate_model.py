import argparse
import json
import os

import cv2
import exifread
import joblib
import numpy as np
import pandas as pd
import pillow_heif
from PIL import Image
from skimage.color import deltaE_ciede2000

pillow_heif.register_heif_opener()

DATASET_ROOT = r"X:\AURORA\processed_photos"
MODEL_PATH = r"X:\AURORA\aurora_model_v1.pkl"


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
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f)

        iso = tags.get("EXIF ISOSpeedRatings", 0)
        exposure = tags.get("EXIF ExposureTime", 0)
        fnumber = tags.get("EXIF FNumber", 0)
        datetime_original = tags.get("EXIF DateTimeOriginal", None)

        if iso != 0:
            iso = int(str(iso))

        if exposure != 0:
            exposure_str = str(exposure)
            if "/" in exposure_str:
                num, den = exposure_str.split("/")
                exposure = float(num) / float(den)
            else:
                exposure = float(exposure_str)

        if fnumber != 0:
            fnumber_str = str(fnumber)
            if "/" in fnumber_str:
                num, den = fnumber_str.split("/")
                fnumber = float(num) / float(den)
            else:
                fnumber = float(fnumber_str)

        hour = 0
        minute = 0

        if datetime_original:
            dt = str(datetime_original)
            time_part = dt.split(" ")[1]
            hour, minute, _ = map(int, time_part.split(":"))

        return iso, exposure, fnumber, hour, minute

    except Exception:
        return 0, 0, 0, 0, 0


def apply_adjustments(rgb, r_gain, b_gain, brightness, contrast):
    arr = rgb.astype(np.float32)
    arr[..., 0] *= float(r_gain)
    arr[..., 2] *= float(b_gain)
    alpha = 1.0 + (float(contrast) / 100.0)
    beta = float(brightness)
    arr = arr * alpha + beta
    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_exposure_rgb(rgb, ev):
    arr = rgb.astype(np.float32) / 255.0
    thr = 0.04045
    lin = np.where(arr <= thr, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    lin *= 2.0 ** float(ev)
    srgb = np.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * (np.clip(lin, 0, 1) ** (1 / 2.4)) - 0.055,
    )
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def parse_folder_tags(folder_name):
    tag_white_walls = 0
    tag_clouds = 0
    tag_no_clouds = 0

    if not isinstance(folder_name, str):
        return tag_white_walls, tag_clouds, tag_no_clouds

    parts = folder_name.split(";")
    if len(parts) < 3:
        return tag_white_walls, tag_clouds, tag_no_clouds

    wall_type = parts[1].strip()
    cloud_type = parts[2].strip()

    if wall_type == "white_walls":
        tag_white_walls = 1

    if cloud_type == "clouds":
        tag_clouds = 1
    elif cloud_type == "no_clouds":
        tag_no_clouds = 1

    return tag_white_walls, tag_clouds, tag_no_clouds


def load_rgb_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))


def get_min_deltae76_roi(meta):
    delta_list = None

    if isinstance(meta.get("deltaE"), dict):
        delta_list = meta["deltaE"].get("per_roi", {}).get("DE76")

    if delta_list is None:
        delta_list = meta.get("deltaE76_per_roi")

    if not delta_list:
        raise ValueError("Missing DE76 ROI list")

    if any(value is None for value in delta_list):
        raise ValueError("DE76 ROI list contains None")

    min_index = int(np.argmin(delta_list))
    rois = meta.get("rois_test")
    if not rois or min_index >= len(rois):
        raise ValueError("Missing ROI coordinates")

    roi = rois[min_index]
    return min_index, int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])


def build_feature_row(meta, folder_name, lab_original, exif_data):
    lux_test = meta.get("lux_test", {})
    lux_min = float(lux_test.get("min", 0))
    lux_max = float(lux_test.get("max", 0))
    lux_mean = (lux_min + lux_max) / 2.0

    iso, exposure, fnumber, hour, minute = exif_data
    L, a, b = lab_original
    tag_white_walls, tag_clouds, tag_no_clouds = parse_folder_tags(folder_name)

    row = {
        "L": float(L),
        "a": float(a),
        "b": float(b),
        "lux_min": lux_min,
        "lux_max": lux_max,
        "lux_mean": lux_mean,
        "ISO": float(iso),
        "exposure_time": float(exposure),
        "f_number": float(fnumber),
        "hour": float(hour),
        "minute": float(minute),
        "tag_white_walls": tag_white_walls,
        "tag_clouds": tag_clouds,
        "tag_no_clouds": tag_no_clouds,
    }

    for tag in meta.get("lighting_tags", []):
        row[f"tag_{str(tag).replace(' ', '_')}"] = 1

    return row


def align_features(feature_row, feature_columns):
    df = pd.DataFrame([feature_row])
    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0
    return df[feature_columns].fillna(0)


def compute_deltae2000(lab_1, lab_2):
    arr_1 = np.array(lab_1, dtype=np.float32).reshape(1, 1, 3)
    arr_2 = np.array(lab_2, dtype=np.float32).reshape(1, 1, 3)
    return float(deltaE_ciede2000(arr_1, arr_2)[0, 0])


def evaluate_sample(sample_root, model_bundle):
    meta_path = os.path.join(sample_root, "meta.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    paths = meta.get("paths", {})
    original_name = paths.get("test_original", "test_original.jpg")
    adjusted_name = paths.get("test_adjusted", "test_adjusted.jpg")

    original_path = os.path.join(sample_root, original_name)
    adjusted_path = os.path.join(sample_root, adjusted_name)

    if not os.path.isfile(original_path):
        raise FileNotFoundError(f"Missing original image: {original_path}")
    if not os.path.isfile(adjusted_path):
        raise FileNotFoundError(f"Missing adjusted image: {adjusted_path}")

    _, x, y, w, h = get_min_deltae76_roi(meta)

    original_rgb = load_rgb_image(original_path)
    adjusted_rgb = load_rgb_image(adjusted_path)

    if y + h > original_rgb.shape[0] or x + w > original_rgb.shape[1]:
        raise ValueError("ROI out of bounds for original image")
    if y + h > adjusted_rgb.shape[0] or x + w > adjusted_rgb.shape[1]:
        raise ValueError("ROI out of bounds for adjusted image")

    roi_original = original_rgb[y:y + h, x:x + w]
    roi_true = adjusted_rgb[y:y + h, x:x + w]

    lab_original = rgb_to_lab_mean(roi_original)
    lab_true = rgb_to_lab_mean(roi_true)

    if lab_original is None or lab_true is None:
        raise ValueError("LAB mean computation failed")

    exif_data = extract_exif_data(original_path)
    folder_name = os.path.basename(os.path.dirname(sample_root))
    feature_row = build_feature_row(meta, folder_name, lab_original, exif_data)

    feature_columns = model_bundle["feature_columns"]
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]

    X = align_features(feature_row, feature_columns)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)[0]

    target_names = model_bundle.get(
        "target_columns",
        ["white_balance", "red_gain", "blue_gain", "brightness", "contrast", "exposure_ev"],
    )
    pred_map = dict(zip(target_names, preds))

    predicted_rgb = apply_adjustments(
        original_rgb,
        pred_map.get("red_gain", 1.0),
        pred_map.get("blue_gain", 1.0),
        pred_map.get("brightness", 0.0),
        pred_map.get("contrast", 0.0),
    )
    predicted_rgb = apply_exposure_rgb(predicted_rgb, pred_map.get("exposure_ev", 0.0))

    roi_predicted = predicted_rgb[y:y + h, x:x + w]
    lab_predicted = rgb_to_lab_mean(roi_predicted)

    if lab_predicted is None:
        raise ValueError("Predicted LAB mean computation failed")

    delta_before = compute_deltae2000(lab_original, lab_true)
    delta_predicted = compute_deltae2000(lab_predicted, lab_true)

    sample_name = os.path.relpath(sample_root, os.path.dirname(DATASET_ROOT))
    return {
        "sample": sample_name,
        "deltaE_before": delta_before,
        "deltaE_predicted": delta_predicted,
    }


def collect_sample_roots(dataset_root):
    sample_roots = []
    for root, _, files in os.walk(dataset_root):
        if "meta.json" in files:
            sample_roots.append(root)
    return sorted(sample_roots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()

    model_bundle = joblib.load(args.model_path)
    sample_roots = collect_sample_roots(args.dataset_root)

    results = []
    skipped = []

    for sample_root in sample_roots:
        try:
            results.append(evaluate_sample(sample_root, model_bundle))
        except Exception as exc:
            skipped.append((sample_root, str(exc)))

    if not results:
        print("Aurora DeltaE Evaluation")
        print("========================")
        print("Samples evaluated: 0")
        print(f"Samples skipped: {len(skipped)}")
        return

    delta_before_values = np.array([row["deltaE_before"] for row in results], dtype=np.float64)
    delta_pred_values = np.array([row["deltaE_predicted"] for row in results], dtype=np.float64)
    improved_mask = delta_pred_values < delta_before_values

    print("Aurora DeltaE Evaluation")
    print("========================")
    print(f"Samples evaluated: {len(results)}")
    print(f"Samples skipped: {len(skipped)}")
    print(f"Mean DeltaE_before: {delta_before_values.mean():.4f}")
    print(f"Mean DeltaE_predicted: {delta_pred_values.mean():.4f}")
    print(f"Improvement: {(delta_before_values.mean() - delta_pred_values.mean()):.4f}")
    print(f"Improved samples: {improved_mask.sum() / len(results) * 100.0:.2f}%")


if __name__ == "__main__":
    main()
