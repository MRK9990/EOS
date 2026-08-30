import argparse
import json
from pathlib import Path

import cv2
import exifread
import joblib
import numpy as np
import pandas as pd
import pillow_heif
from PIL import Image
from skimage.color import deltaE_ciede2000


pillow_heif.register_heif_opener()


# ============================================================
# EOS PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "processed_photos"
MODEL_PATH = PROJECT_ROOT / "models" / "eos_model_v1.pkl"


# ============================================================
# IMAGE / COLOR FUNCTIONS
# ============================================================

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


def apply_adjustments(
    rgb,
    r_gain,
    b_gain,
    brightness,
    contrast
):
    arr = rgb.astype(np.float32)

    arr[..., 0] *= float(r_gain)
    arr[..., 2] *= float(b_gain)

    alpha = 1.0 + (float(contrast) / 100.0)
    beta = float(brightness)

    arr = arr * alpha + beta

    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_exposure_rgb(rgb, ev):
    arr = rgb.astype(np.float32) / 255.0

    threshold = 0.04045

    linear = np.where(
        arr <= threshold,
        arr / 12.92,
        ((arr + 0.055) / 1.055) ** 2.4
    )

    linear *= 2.0 ** float(ev)

    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (np.clip(linear, 0, 1) ** (1 / 2.4)) - 0.055
    )

    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# DATASET METADATA
# ============================================================

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

    return (
        tag_white_walls,
        tag_clouds,
        tag_no_clouds
    )


def load_rgb_image(image_path):
    return np.array(
        Image.open(image_path).convert("RGB")
    )


def get_min_deltae76_roi(meta):
    delta_list = None

    if isinstance(meta.get("deltaE"), dict):
        delta_list = (
            meta["deltaE"]
            .get("per_roi", {})
            .get("DE76")
        )

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

    return (
        min_index,
        int(roi["x"]),
        int(roi["y"]),
        int(roi["w"]),
        int(roi["h"])
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_row(
    meta,
    folder_name,
    lab_original,
    exif_data
):
    lux_test = meta.get("lux_test", {})

    lux_min = float(
        lux_test.get("min", 0)
    )

    lux_max = float(
        lux_test.get("max", 0)
    )

    lux_mean = (lux_min + lux_max) / 2.0

    iso, exposure, fnumber, hour, minute = exif_data

    L, a, b = lab_original

    (
        tag_white_walls,
        tag_clouds,
        tag_no_clouds
    ) = parse_folder_tags(folder_name)

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
        row[
            f"tag_{str(tag).replace(' ', '_')}"
        ] = 1

    return row


def align_features(feature_row, feature_columns):
    df = pd.DataFrame([feature_row])

    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0

    return df[feature_columns].fillna(0)


# ============================================================
# DELTA EVALUATION
# ============================================================

def compute_deltae2000(lab_1, lab_2):
    arr_1 = np.array(
        lab_1,
        dtype=np.float32
    ).reshape(1, 1, 3)

    arr_2 = np.array(
        lab_2,
        dtype=np.float32
    ).reshape(1, 1, 3)

    return float(
        deltaE_ciede2000(
            arr_1,
            arr_2
        )[0, 0]
    )


# ============================================================
# SAMPLE EVALUATION
# ============================================================

def evaluate_sample(
    sample_root,
    model_bundle
):
    sample_root = Path(sample_root)

    meta_path = sample_root / "meta.json"

    with open(
        meta_path,
        "r",
        encoding="utf-8"
    ) as f:
        meta = json.load(f)

    paths = meta.get("paths", {})

    original_name = paths.get(
        "test_original",
        "test_original.jpg"
    )

    adjusted_name = paths.get(
        "test_adjusted",
        "test_adjusted.jpg"
    )

    original_path = sample_root / original_name
    adjusted_path = sample_root / adjusted_name

    if not original_path.is_file():
        raise FileNotFoundError(
            f"Missing original image: {original_path}"
        )

    if not adjusted_path.is_file():
        raise FileNotFoundError(
            f"Missing adjusted image: {adjusted_path}"
        )

    _, x, y, w, h = get_min_deltae76_roi(meta)

    original_rgb = load_rgb_image(
        original_path
    )

    adjusted_rgb = load_rgb_image(
        adjusted_path
    )

    if (
        y + h > original_rgb.shape[0]
        or x + w > original_rgb.shape[1]
    ):
        raise ValueError(
            "ROI out of bounds for original image"
        )

    if (
        y + h > adjusted_rgb.shape[0]
        or x + w > adjusted_rgb.shape[1]
    ):
        raise ValueError(
            "ROI out of bounds for adjusted image"
        )

    roi_original = original_rgb[
        y:y + h,
        x:x + w
    ]

    roi_true = adjusted_rgb[
        y:y + h,
        x:x + w
    ]

    lab_original = rgb_to_lab_mean(
        roi_original
    )

    lab_true = rgb_to_lab_mean(
        roi_true
    )

    if lab_original is None:
        raise ValueError(
            "LAB mean computation failed for original image"
        )

    if lab_true is None:
        raise ValueError(
            "LAB mean computation failed for adjusted image"
        )

    exif_data = extract_exif_data(
        original_path
    )

    folder_name = sample_root.parent.name

    feature_row = build_feature_row(
        meta,
        folder_name,
        lab_original,
        exif_data
    )

    feature_columns = model_bundle[
        "feature_columns"
    ]

    scaler = model_bundle["scaler"]
    model = model_bundle["model"]

    X = align_features(
        feature_row,
        feature_columns
    )

    X_scaled = scaler.transform(X)

    predictions = model.predict(
        X_scaled
    )[0]

    target_names = model_bundle.get(
        "target_columns",
        [
            "white_balance",
            "red_gain",
            "blue_gain",
            "brightness",
            "contrast",
            "exposure_ev"
        ]
    )

    pred_map = dict(
        zip(target_names, predictions)
    )

    predicted_rgb = apply_adjustments(
        original_rgb,
        pred_map.get(
            "red_gain",
            1.0
        ),
        pred_map.get(
            "blue_gain",
            1.0
        ),
        pred_map.get(
            "brightness",
            0.0
        ),
        pred_map.get(
            "contrast",
            0.0
        )
    )

    predicted_rgb = apply_exposure_rgb(
        predicted_rgb,
        pred_map.get(
            "exposure_ev",
            0.0
        )
    )

    roi_predicted = predicted_rgb[
        y:y + h,
        x:x + w
    ]

    lab_predicted = rgb_to_lab_mean(
        roi_predicted
    )

    if lab_predicted is None:
        raise ValueError(
            "Predicted LAB mean computation failed"
        )

    delta_before = compute_deltae2000(
        lab_original,
        lab_true
    )

    delta_predicted = compute_deltae2000(
        lab_predicted,
        lab_true
    )

    return {
        "sample": sample_root.name,
        "deltaE_before": delta_before,
        "deltaE_predicted": delta_predicted,
    }


# ============================================================
# DATASET DISCOVERY
# ============================================================

def collect_sample_roots(dataset_root):
    dataset_root = Path(dataset_root)

    sample_roots = []

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}"
        )

    for meta_path in dataset_root.rglob(
        "meta.json"
    ):
        sample_roots.append(
            meta_path.parent
        )

    return sorted(sample_roots)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EOS model using DeltaE2000."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=(
            "Path to processed EOS dataset "
            "(default: data/processed_photos)"
        )
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=(
            "Path to trained EOS model "
            "(default: models/eos_model_v1.pkl)"
        )
    )

    args = parser.parse_args()

    print("EOS DeltaE Evaluation")
    print("=====================")

    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model_path}"
        )

    print(
        f"Model:   {args.model_path}"
    )

    print(
        f"Dataset: {args.dataset_root}"
    )

    model_bundle = joblib.load(
        args.model_path
    )

    sample_roots = collect_sample_roots(
        args.dataset_root
    )

    print(
        f"Samples found: {len(sample_roots)}"
    )

    results = []
    skipped = []

    for sample_root in sample_roots:

        try:
            result = evaluate_sample(
                sample_root,
                model_bundle
            )

            results.append(result)

        except Exception as exc:

            skipped.append(
                (
                    str(sample_root),
                    str(exc)
                )
            )

    print()

    if not results:
        print("No samples were successfully evaluated.")
        print(
            f"Samples skipped: {len(skipped)}"
        )
        return

    delta_before_values = np.array(
        [
            row["deltaE_before"]
            for row in results
        ],
        dtype=np.float64
    )

    delta_predicted_values = np.array(
        [
            row["deltaE_predicted"]
            for row in results
        ],
        dtype=np.float64
    )

    improved_mask = (
        delta_predicted_values
        < delta_before_values
    )

    mean_before = (
        delta_before_values.mean()
    )

    mean_predicted = (
        delta_predicted_values.mean()
    )

    improvement = (
        mean_before
        - mean_predicted
    )

    improvement_percent = (
        improvement / mean_before * 100.0
        if mean_before != 0
        else 0.0
    )

    print("Results")
    print("=======")

    print(
        f"Samples evaluated: "
        f"{len(results)}"
    )

    print(
        f"Samples skipped: "
        f"{len(skipped)}"
    )

    print(
        f"Mean DeltaE before: "
        f"{mean_before:.4f}"
    )

    print(
        f"Mean DeltaE predicted: "
        f"{mean_predicted:.4f}"
    )

    print(
        f"Absolute improvement: "
        f"{improvement:.4f}"
    )

    print(
        f"Relative improvement: "
        f"{improvement_percent:.2f}%"
    )

    print(
        f"Improved samples: "
        f"{improved_mask.sum()} / "
        f"{len(results)} "
        f"({improved_mask.mean() * 100:.2f}%)"
    )

    if skipped:
        print()
        print("Skipped samples")
        print("===============")

        for sample_path, error in skipped[:10]:
            print(
                f"- {sample_path}: {error}"
            )

        if len(skipped) > 10:
            print(
                f"... and "
                f"{len(skipped) - 10} more."
            )


if __name__ == "__main__":
    main()