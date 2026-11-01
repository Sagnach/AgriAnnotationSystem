from __future__ import annotations

from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from importlib import metadata as importlib_metadata

import argparse
import csv
import json
import math
import platform
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
from PIL import Image


# =============================================================================
# SAM 3 Zero-Shot Baseline on MinneApple
# =============================================================================
#
# Purpose
# -------
# 1. Read the frozen MinneApple manifest created by 02_prepare_dataset.py.
# 2. Run SAM 3 zero-shot concept segmentation with prompt="apple".
# 3. Cache every prediction so the experiment can resume after interruption.
# 4. Store masks compactly as COCO RLE rather than large PNG/NPZ arrays.
# 5. Evaluate predictions against indexed MinneApple instance masks.
# 6. Sweep confidence thresholds WITHOUT rerunning SAM 3.
# 7. Report Precision / Recall / F1 / matched mask IoU / AutoAccept@IoU.
# 8. Compute score-ranked mask AP50 and AP75 from the cached predictions.
# 9. Save per-image metrics, threshold sweep CSV, plots, and summary JSON.
#
# Default experiment
# ------------------
#   Split            : validation only
#   Prompt           : apple
#   SAM 3 base score : 0.25  (cache low-score candidates once)
#   Sweep            : 0.30 ... 0.90
#   TP criterion     : mask IoU >= 0.50
#   Auto-accept      : mask IoU >= 0.85
#
# IMPORTANT
# ---------
# The official MinneApple test split is frozen. This script refuses to run on
# split=test unless --allow-test is supplied explicitly.
#
# =============================================================================


MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)
MIN_CUDA = (12, 6)

DEFAULT_PROMPT = "apple"
DEFAULT_BASE_CONFIDENCE = 0.25
DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
DEFAULT_MATCH_IOU = 0.50
DEFAULT_AUTO_ACCEPT_IOU = 0.85

MANIFEST_RELATIVE_PATH = Path("data/splits/minneapple_split.csv")
DEFAULT_CHECKPOINT_RELATIVE_PATH = Path("models/sam3/checkpoints/sam3.pt")
THIRD_PARTY_SAM3_RELATIVE_PATH = Path("third_party/sam3")
OUTPUTS_RELATIVE_PATH = Path("outputs/experiments")


# =============================================================================
# Generic utilities
# =============================================================================


def version_tuple(text: str | None, length: int = 3) -> tuple[int, ...]:
    if not text:
        return tuple(0 for _ in range(length))

    values: list[int] = []
    current = ""
    for char in str(text):
        if char.isdigit():
            current += char
        else:
            if current:
                values.append(int(current))
                current = ""
                if len(values) >= length:
                    break

    if current and len(values) < length:
        values.append(int(current))

    while len(values) < length:
        values.append(0)

    return tuple(values[:length])


def human_bytes(num_bytes: int | float | None) -> str | None:
    if num_bytes is None:
        return None

    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def safe_package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def resolve_project_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    # Expected path:
    # AgriAnnotationSystem/scripts/04_run_zero_shot_baseline.py
    return Path(__file__).resolve().parents[1]


def resolve_manifest_path(project_root: Path, explicit_manifest: str | None) -> Path:
    if explicit_manifest:
        path = Path(explicit_manifest).expanduser()
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()

    return (project_root / MANIFEST_RELATIVE_PATH).resolve()


def resolve_row_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def relative_or_absolute(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_thresholds(values: list[float] | None) -> list[float]:
    raw = values if values else list(DEFAULT_THRESHOLDS)
    thresholds = sorted(set(round(float(v), 6) for v in raw))
    for value in thresholds:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid confidence threshold: {value}")
    if not thresholds:
        raise ValueError("At least one confidence threshold is required.")
    return thresholds


# =============================================================================
# Environment / SAM 3 loading
# =============================================================================


def collect_environment() -> tuple[dict, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    environment: dict = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    if sys.version_info[:2] < MIN_PYTHON:
        errors.append(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is recommended; "
            f"found {platform.python_version()}."
        )

    try:
        import torch

        environment["torch_version"] = torch.__version__
        environment["torchvision_version"] = safe_package_version("torchvision")
        environment["torch_cuda_version"] = torch.version.cuda
        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["cudnn_version"] = (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            and torch.backends.cudnn.version() is not None
            else None
        )

        if version_tuple(torch.__version__, 2) < MIN_TORCH:
            errors.append(
                f"PyTorch {MIN_TORCH[0]}.{MIN_TORCH[1]}+ is recommended; "
                f"found {torch.__version__}."
            )

        if torch.version.cuda is not None and version_tuple(torch.version.cuda, 2) < MIN_CUDA:
            errors.append(
                f"CUDA {MIN_CUDA[0]}.{MIN_CUDA[1]}+ is recommended; "
                f"PyTorch reports CUDA {torch.version.cuda}."
            )

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            environment.update(
                {
                    "cuda_device_index": int(device_index),
                    "cuda_device_name": torch.cuda.get_device_name(device_index),
                    "cuda_compute_capability": f"{props.major}.{props.minor}",
                    "cuda_total_memory_bytes": int(props.total_memory),
                    "cuda_total_memory": human_bytes(props.total_memory),
                    "bf16_supported": bool(
                        torch.cuda.is_bf16_supported()
                        if hasattr(torch.cuda, "is_bf16_supported")
                        else False
                    ),
                }
            )
        else:
            warnings.append("PyTorch cannot access a CUDA GPU.")

    except Exception as error:
        environment["torch_import_error"] = repr(error)
        errors.append(f"PyTorch import failed: {error}")

    return environment, errors, warnings


def add_local_sam3_to_path(project_root: Path) -> Path | None:
    repo = (project_root / THIRD_PARTY_SAM3_RELATIVE_PATH).resolve()
    if (repo / "sam3" / "model_builder.py").is_file():
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        return repo
    return None


def import_sam3(project_root: Path):
    local_repo = add_local_sam3_to_path(project_root)

    try:
        import sam3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as error:
        raise RuntimeError(
            "SAM 3 could not be imported. Ensure the official repository is installed "
            "in the active Python environment and all inference dependencies are present."
        ) from error

    package_info = {
        "sam3_version": getattr(sam3, "__version__", safe_package_version("sam3")),
        "sam3_module_file": getattr(sam3, "__file__", None),
        "local_repo_detected": str(local_repo) if local_repo else None,
    }

    return build_sam3_image_model, Sam3Processor, package_info


def resolve_checkpoint(project_root: Path, explicit_checkpoint: str | None) -> Path | None:
    if explicit_checkpoint:
        path = Path(explicit_checkpoint).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    preferred = (project_root / DEFAULT_CHECKPOINT_RELATIVE_PATH).resolve()
    if preferred.is_file():
        return preferred

    checkpoint_dir = preferred.parent
    if checkpoint_dir.is_dir():
        candidates = sorted(checkpoint_dir.glob("*.pt"))
        sam3_candidates = [
            p for p in candidates if "sam3" in p.name.lower() and "3.1" not in p.name.lower()
        ]
        if len(sam3_candidates) == 1:
            return sam3_candidates[0].resolve()

    return None


# =============================================================================
# Manifest / GT handling
# =============================================================================


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen MinneApple manifest not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    required = {
        "sample_id",
        "split",
        "instance_count",
        "processed_image_path",
        "processed_mask_path",
    }
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {sorted(missing)}")

    return rows


def select_split_rows(
    rows: list[dict[str, str]],
    split: str,
    limit: int | None,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("split", "").strip().lower() == split]
    selected = sorted(selected, key=lambda row: row.get("sample_id", ""))

    if not selected:
        raise RuntimeError(f"No rows found for split={split!r}.")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        selected = selected[:limit]

    return selected


def load_gt_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {path}")

    with Image.open(path) as image:
        mask = np.array(image)

    if mask.ndim != 2:
        raise RuntimeError(f"Expected 2-D indexed GT mask; found shape={mask.shape}")

    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]
    return mask, instance_ids


# =============================================================================
# Tensor / prediction conversion
# =============================================================================


def _torch_tensor_to_numpy(value) -> np.ndarray:
    import torch

    tensor = value.detach()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(dtype=torch.float32)
    elif str(tensor.dtype).startswith("torch.float8"):
        tensor = tensor.to(dtype=torch.float32)
    tensor = tensor.to(device="cpu")
    return tensor.numpy()


def tensor_to_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)

    import torch

    if isinstance(value, torch.Tensor):
        return _torch_tensor_to_numpy(value)

    if isinstance(value, (list, tuple)):
        converted = [tensor_to_numpy(item) for item in value]
        if not converted:
            return np.empty((0,), dtype=np.float32)
        try:
            return np.stack(converted, axis=0)
        except ValueError:
            return np.asarray(converted, dtype=object)

    return np.asarray(value)


def normalize_prediction_masks(
    masks,
    expected_height: int,
    expected_width: int,
) -> np.ndarray:
    if masks is None:
        return np.zeros((0, expected_height, expected_width), dtype=bool)

    masks = tensor_to_numpy(masks)
    if masks.size == 0:
        return np.zeros((0, expected_height, expected_width), dtype=bool)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]

    if masks.ndim != 3:
        raise RuntimeError(f"Unexpected SAM 3 mask shape: {masks.shape}")

    if masks.shape[1:] != (expected_height, expected_width):
        raise RuntimeError(
            "SAM 3 mask size does not match the original image: "
            f"masks={masks.shape[1:]}, image={(expected_height, expected_width)}"
        )

    return masks.astype(bool, copy=False)


# =============================================================================
# COCO RLE cache helpers
# =============================================================================


def encode_binary_mask(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")

    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts,
    }


def decode_binary_mask(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_utils

    encoded = {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].encode("ascii")
        if isinstance(rle["counts"], str)
        else rle["counts"],
    }
    mask = mask_utils.decode(encoded)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


# =============================================================================
# Prediction cache
# =============================================================================


def cache_signature(
    sample: dict[str, str],
    prompt: str,
    base_confidence: float,
) -> dict:
    return {
        "sample_id": sample.get("sample_id"),
        "image_path": sample.get("processed_image_path"),
        "mask_path": sample.get("processed_mask_path"),
        "prompt": prompt,
        "base_confidence": round(float(base_confidence), 6),
    }


def cache_is_compatible(data: dict, expected_signature: dict) -> bool:
    return data.get("signature") == expected_signature and data.get("status") == "PASS"


def prediction_cache_path(prediction_dir: Path, sample_id: str) -> Path:
    return prediction_dir / f"{sample_id}.json"


def save_prediction_cache(
    path: Path,
    signature: dict,
    sample: dict[str, str],
    image_size: tuple[int, int],
    gt_instances: int,
    predictions: list[dict],
    inference_seconds: float,
    autocast_label: str,
) -> None:
    payload = {
        "cache_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "status": "PASS",
        "signature": signature,
        "sample": {
            "sample_id": sample.get("sample_id"),
            "split": sample.get("split"),
            "group_id": sample.get("group_id"),
            "image_path": sample.get("processed_image_path"),
            "mask_path": sample.get("processed_mask_path"),
            "width": int(image_size[0]),
            "height": int(image_size[1]),
            "ground_truth_instances": int(gt_instances),
        },
        "inference": {
            "seconds": round(float(inference_seconds), 6),
            "autocast": autocast_label,
            "predicted_instances": len(predictions),
        },
        "predictions": predictions,
    }
    write_json(path, payload)


def run_sam3_inference_one(
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    use_autocast: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    import torch

    width, height = image.size

    if use_autocast:
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        autocast_label = "cuda/bfloat16"
    else:
        autocast_context = nullcontext()
        autocast_label = "disabled"

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        with autocast_context:
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=prompt)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    if not isinstance(output, dict):
        raise RuntimeError(f"SAM 3 returned unexpected output type: {type(output)}")

    masks = normalize_prediction_masks(output.get("masks"), height, width)
    boxes = tensor_to_numpy(output.get("boxes"))
    scores = tensor_to_numpy(output.get("scores")).reshape(-1)

    boxes = np.empty((0, 4), dtype=np.float32) if boxes.size == 0 else np.asarray(boxes).reshape(-1, 4)

    if len(masks) != len(boxes) or len(masks) != len(scores):
        raise RuntimeError(
            "SAM 3 prediction count mismatch: "
            f"masks={len(masks)}, boxes={len(boxes)}, scores={len(scores)}"
        )

    return masks, boxes, scores, elapsed, autocast_label


def build_prediction_records(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_label: str,
) -> list[dict]:
    records: list[dict] = []

    for index in range(len(masks)):
        mask = masks[index]
        box = boxes[index]
        records.append(
            {
                "instance_index": index + 1,
                "class_label": class_label,
                "score": float(scores[index]),
                "box_xyxy": [float(v) for v in box[:4]],
                "mask_area_pixels": int(mask.sum()),
                "rle": encode_binary_mask(mask),
            }
        )

    return records


# =============================================================================
# IoU / matching
# =============================================================================


def compute_iou_matrix(
    gt_indexed_mask: np.ndarray,
    gt_ids: np.ndarray,
    pred_masks: list[np.ndarray],
) -> np.ndarray:
    """Efficient prediction x GT mask-IoU matrix.

    Instead of performing P x G full-resolution logical operations, each
    prediction uses np.bincount on the GT labels under its foreground pixels.
    """
    num_pred = len(pred_masks)
    num_gt = len(gt_ids)

    if num_pred == 0 or num_gt == 0:
        return np.zeros((num_pred, num_gt), dtype=np.float32)

    max_label = int(max(int(gt_indexed_mask.max()), int(gt_ids.max(initial=0))))
    gt_area_all = np.bincount(gt_indexed_mask.ravel().astype(np.int64), minlength=max_label + 1)
    gt_areas = gt_area_all[gt_ids.astype(np.int64)].astype(np.float64)

    iou = np.zeros((num_pred, num_gt), dtype=np.float32)

    for pred_index, pred_mask in enumerate(pred_masks):
        pred_mask = pred_mask.astype(bool, copy=False)
        pred_area = int(pred_mask.sum())
        if pred_area == 0:
            continue

        labels_under_pred = gt_indexed_mask[pred_mask].astype(np.int64, copy=False)
        intersections_all = np.bincount(labels_under_pred, minlength=max_label + 1)
        intersections = intersections_all[gt_ids.astype(np.int64)].astype(np.float64)
        unions = pred_area + gt_areas - intersections

        valid = unions > 0
        row = np.zeros(num_gt, dtype=np.float32)
        row[valid] = (intersections[valid] / unions[valid]).astype(np.float32)
        iou[pred_index] = row

    return iou


def match_score_order(
    iou_matrix: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict:
    """Standard score-ordered one-to-one matching for one image."""
    num_pred, num_gt = iou_matrix.shape

    selected = np.where(scores >= confidence_threshold)[0]
    selected = selected[np.argsort(scores[selected])[::-1]] if len(selected) else selected

    matched_gt: set[int] = set()
    matches: list[dict] = []
    tp = 0
    fp = 0

    for pred_index in selected:
        if num_gt == 0:
            fp += 1
            matches.append(
                {
                    "pred_index": int(pred_index),
                    "gt_index": None,
                    "iou": 0.0,
                    "score": float(scores[pred_index]),
                    "is_tp": False,
                }
            )
            continue

        row = iou_matrix[pred_index]
        candidate_order = np.argsort(row)[::-1]
        chosen_gt = None
        chosen_iou = 0.0

        for gt_index in candidate_order:
            gt_index = int(gt_index)
            if gt_index in matched_gt:
                continue
            chosen_gt = gt_index
            chosen_iou = float(row[gt_index])
            break

        if chosen_gt is not None and chosen_iou >= iou_threshold:
            matched_gt.add(chosen_gt)
            tp += 1
            is_tp = True
        else:
            fp += 1
            is_tp = False

        matches.append(
            {
                "pred_index": int(pred_index),
                "gt_index": int(chosen_gt) if is_tp and chosen_gt is not None else None,
                "iou": chosen_iou,
                "score": float(scores[pred_index]),
                "is_tp": is_tp,
            }
        )

    fn = num_gt - tp
    matched_ious = [item["iou"] for item in matches if item["is_tp"]]

    return {
        "selected_predictions": int(len(selected)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "matches": matches,
        "matched_ious": matched_ious,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


# =============================================================================
# Dataset evaluation
# =============================================================================


def load_eval_record(
    project_root: Path,
    sample: dict[str, str],
    cache_data: dict,
) -> dict:
    mask_path = resolve_row_path(project_root, sample["processed_mask_path"])
    gt_mask, gt_ids = load_gt_mask(mask_path)

    predictions = cache_data.get("predictions", [])
    scores = np.asarray([float(item["score"]) for item in predictions], dtype=np.float32)
    pred_masks = [decode_binary_mask(item["rle"]) for item in predictions]

    iou_matrix = compute_iou_matrix(gt_mask, gt_ids, pred_masks)

    return {
        "sample_id": sample["sample_id"],
        "gt_count": int(len(gt_ids)),
        "pred_count": int(len(predictions)),
        "scores": scores,
        "iou_matrix": iou_matrix,
        "inference_seconds": float(cache_data.get("inference", {}).get("seconds", 0.0)),
    }


def evaluate_threshold(
    records: list[dict],
    confidence_threshold: float,
    match_iou: float,
    auto_accept_iou: float,
) -> tuple[dict, list[dict]]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    selected_predictions = 0
    matched_ious: list[float] = []

    auto_tp = 0
    total_gt = 0

    per_image_rows: list[dict] = []

    for record in records:
        standard = match_score_order(
            record["iou_matrix"],
            record["scores"],
            confidence_threshold,
            match_iou,
        )
        accept = match_score_order(
            record["iou_matrix"],
            record["scores"],
            confidence_threshold,
            auto_accept_iou,
        )

        total_tp += standard["tp"]
        total_fp += standard["fp"]
        total_fn += standard["fn"]
        selected_predictions += standard["selected_predictions"]
        matched_ious.extend(standard["matched_ious"])
        auto_tp += accept["tp"]
        total_gt += record["gt_count"]

        precision, recall, f1 = precision_recall_f1(
            standard["tp"], standard["fp"], standard["fn"]
        )

        per_image_rows.append(
            {
                "sample_id": record["sample_id"],
                "confidence_threshold": confidence_threshold,
                "gt_instances": record["gt_count"],
                "cached_predictions": record["pred_count"],
                "kept_predictions": standard["selected_predictions"],
                "tp_iou50": standard["tp"],
                "fp_iou50": standard["fp"],
                "fn_iou50": standard["fn"],
                "precision_iou50": round(precision, 6),
                "recall_iou50": round(recall, 6),
                "f1_iou50": round(f1, 6),
                "mean_matched_iou": round(float(np.mean(standard["matched_ious"])), 6)
                if standard["matched_ious"]
                else 0.0,
                "auto_accept_tp": accept["tp"],
                "auto_accept_rate_gt": round(safe_divide(accept["tp"], record["gt_count"]), 6),
                "inference_seconds": round(record["inference_seconds"], 6),
            }
        )

    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    metrics = {
        "confidence_threshold": float(confidence_threshold),
        "matching_iou_threshold": float(match_iou),
        "auto_accept_iou_threshold": float(auto_accept_iou),
        "images": len(records),
        "ground_truth_instances": int(total_gt),
        "kept_predictions": int(selected_predictions),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_matched_iou,
        "auto_accept_tp": int(auto_tp),
        "auto_accept_rate_gt": safe_divide(auto_tp, total_gt),
        "mean_predictions_per_image": safe_divide(selected_predictions, len(records)),
    }

    return metrics, per_image_rows


# =============================================================================
# Score-ranked AP
# =============================================================================


def compute_average_precision(records: list[dict], iou_threshold: float) -> dict:
    """101-point interpolated AP using cached predictions sorted by score."""
    total_gt = sum(int(record["gt_count"]) for record in records)
    if total_gt == 0:
        return {
            "iou_threshold": float(iou_threshold),
            "ap": 0.0,
            "total_gt": 0,
            "total_predictions": 0,
        }

    candidates: list[tuple[float, int, int]] = []
    for image_index, record in enumerate(records):
        for pred_index, score in enumerate(record["scores"]):
            candidates.append((float(score), image_index, pred_index))

    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_gt_by_image: dict[int, set[int]] = defaultdict(set)
    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for score, image_index, pred_index in candidates:
        record = records[image_index]
        iou_row = record["iou_matrix"][pred_index]
        matched_gt = matched_gt_by_image[image_index]

        if len(iou_row) == 0:
            tp_flags.append(0)
            fp_flags.append(1)
            continue

        order = np.argsort(iou_row)[::-1]
        chosen_gt = None
        chosen_iou = 0.0

        for gt_index in order:
            gt_index = int(gt_index)
            if gt_index in matched_gt:
                continue
            chosen_gt = gt_index
            chosen_iou = float(iou_row[gt_index])
            break

        if chosen_gt is not None and chosen_iou >= iou_threshold:
            matched_gt.add(chosen_gt)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    if not candidates:
        return {
            "iou_threshold": float(iou_threshold),
            "ap": 0.0,
            "total_gt": int(total_gt),
            "total_predictions": 0,
        }

    tp_cum = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fp_flags, dtype=np.float64))

    recall = tp_cum / float(total_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated = []
    for recall_level in recall_grid:
        valid = recall >= recall_level
        interpolated.append(float(np.max(precision[valid])) if np.any(valid) else 0.0)

    ap = float(np.mean(interpolated))

    return {
        "iou_threshold": float(iou_threshold),
        "ap": ap,
        "total_gt": int(total_gt),
        "total_predictions": int(len(candidates)),
        "max_recall": float(recall[-1]) if len(recall) else 0.0,
        "final_precision": float(precision[-1]) if len(precision) else 0.0,
    }


# =============================================================================
# Plot
# =============================================================================


def save_threshold_plot(path: Path, sweep_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    x = [row["confidence_threshold"] for row in sweep_rows]
    precision = [row["precision"] for row in sweep_rows]
    recall = [row["recall"] for row in sweep_rows]
    f1 = [row["f1"] for row in sweep_rows]
    auto_accept = [row["auto_accept_rate_gt"] for row in sweep_rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, precision, marker="o", label="Precision @ IoU>=0.50")
    ax.plot(x, recall, marker="o", label="Recall @ IoU>=0.50")
    ax.plot(x, f1, marker="o", label="F1 @ IoU>=0.50")
    ax.plot(x, auto_accept, marker="o", label="AutoAccept rate @ IoU>=0.85")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("SAM 3 Zero-Shot MinneApple Validation Threshold Sweep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SAM 3 zero-shot MinneApple baseline with resumable prediction caching."
    )

    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--base-confidence", type=float, default=DEFAULT_BASE_CONFIDENCE)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--auto-accept-iou", type=float, default=DEFAULT_AUTO_ACCEPT_IOU)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore compatible prediction caches and rerun SAM 3 for every selected image.",
    )

    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)

    if not 0.0 <= args.base_confidence <= 1.0:
        raise SystemExit("--base-confidence must be between 0 and 1.")
    if min(thresholds) < args.base_confidence:
        raise SystemExit(
            "The smallest sweep threshold cannot be below --base-confidence because "
            "predictions below the SAM 3 inference threshold are not cached."
        )
    if not 0.0 <= args.match_iou <= 1.0:
        raise SystemExit("--match-iou must be between 0 and 1.")
    if not 0.0 <= args.auto_accept_iou <= 1.0:
        raise SystemExit("--auto-accept-iou must be between 0 and 1.")
    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "The official test split is frozen. Use validation for threshold selection. "
            "Pass --allow-test only after the validation protocol is frozen."
        )

    project_root = resolve_project_root(args.project_root)
    manifest_path = resolve_manifest_path(project_root, args.manifest)

    experiment_name = f"sam3_zero_shot_{args.split}"
    if args.limit is not None:
        experiment_name += f"_limit{args.limit}"

    experiment_root = (project_root / OUTPUTS_RELATIVE_PATH / experiment_name).resolve()
    prediction_dir = experiment_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    summary_path = experiment_root / "baseline_summary.json"
    sweep_csv = experiment_root / "threshold_sweep.csv"
    per_image_csv = experiment_root / "per_image_metrics_best_threshold.csv"
    cache_index_csv = experiment_root / "prediction_cache_index.csv"
    plot_path = experiment_root / "threshold_sweep.png"

    summary: dict = {
        "baseline_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "status": "RUNNING",
        "project_root": str(project_root),
        "manifest": relative_or_absolute(manifest_path, project_root),
        "split": args.split,
        "prompt": args.prompt,
        "base_confidence": args.base_confidence,
        "thresholds": thresholds,
        "matching_iou": args.match_iou,
        "auto_accept_iou": args.auto_accept_iou,
        "errors": [],
        "warnings": [],
    }

    print()
    print("=" * 88)
    print("SAM 3 ZERO-SHOT BASELINE — MINNEAPPLE")
    print("=" * 88)
    print(f"Project root        : {project_root}")
    print(f"Manifest            : {manifest_path}")
    print(f"Split               : {args.split}")
    print(f"Prompt              : {args.prompt!r}")
    print(f"Base confidence     : {args.base_confidence:.2f}")
    print(f"Threshold sweep     : {thresholds}")
    print(f"TP matching IoU     : {args.match_iou:.2f}")
    print(f"AutoAccept IoU      : {args.auto_accept_iou:.2f}")
    print(f"Experiment output   : {experiment_root}")
    print()

    try:
        # ------------------------------------------------------------------
        # 1/6 Environment + manifest
        # ------------------------------------------------------------------
        print("[1/6] Auditing environment and loading frozen manifest...")
        environment, env_errors, env_warnings = collect_environment()
        summary["environment"] = environment
        summary["errors"].extend(env_errors)
        summary["warnings"].extend(env_warnings)

        if env_errors:
            raise RuntimeError("Environment audit failed:\n  - " + "\n  - ".join(env_errors))

        import torch

        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
        if device == "cpu" and not args.allow_cpu:
            raise RuntimeError(
                "CPU inference is disabled by default. Use CUDA or pass --allow-cpu explicitly."
            )

        summary["device"] = device

        rows = load_manifest(manifest_path)
        selected_rows = select_split_rows(rows, args.split, args.limit)
        summary["selected_images"] = len(selected_rows)
        summary["selected_gt_instances_manifest"] = int(
            sum(int(float(row.get("instance_count", "0") or 0)) for row in selected_rows)
        )

        print(f"  Device              : {device}")
        print(f"  Selected images     : {len(selected_rows)}")
        print(f"  Manifest GT apples  : {summary['selected_gt_instances_manifest']}")

        # ------------------------------------------------------------------
        # 2/6 Inspect caches, determine pending inference
        # ------------------------------------------------------------------
        print()
        print("[2/6] Inspecting resumable prediction cache...")

        cached_data_by_sample: dict[str, dict] = {}
        pending_rows: list[dict[str, str]] = []
        cache_index_rows: list[dict] = []

        for row in selected_rows:
            sample_id = row["sample_id"]
            cache_path = prediction_cache_path(prediction_dir, sample_id)
            signature = cache_signature(row, args.prompt, args.base_confidence)

            cache_state = "missing"
            data = None

            if cache_path.is_file() and not args.overwrite:
                try:
                    data = read_json(cache_path)
                    if cache_is_compatible(data, signature):
                        cache_state = "reused"
                        cached_data_by_sample[sample_id] = data
                    else:
                        cache_state = "stale"
                        pending_rows.append(row)
                except Exception:
                    cache_state = "invalid"
                    pending_rows.append(row)
            else:
                pending_rows.append(row)
                if cache_path.is_file() and args.overwrite:
                    cache_state = "overwrite"

            cache_index_rows.append(
                {
                    "sample_id": sample_id,
                    "split": row.get("split"),
                    "cache_state": cache_state,
                    "cache_path": relative_or_absolute(cache_path, project_root),
                }
            )

        print(f"  Reusable caches     : {len(cached_data_by_sample)}")
        print(f"  Pending inference   : {len(pending_rows)}")

        # ------------------------------------------------------------------
        # 3/6 Run missing SAM 3 inference
        # ------------------------------------------------------------------
        print()
        print("[3/6] Running SAM 3 inference for uncached images...")

        model_load_seconds = 0.0
        checkpoint_info = None
        package_info = None
        peak_memory_bytes = None

        if pending_rows:
            build_sam3_image_model, Sam3Processor, package_info = import_sam3(project_root)
            checkpoint_path = resolve_checkpoint(project_root, args.checkpoint)

            if checkpoint_path is None and not args.allow_hf_download:
                expected = project_root / DEFAULT_CHECKPOINT_RELATIVE_PATH
                raise RuntimeError(
                    "No local SAM 3 checkpoint was found.\n"
                    f"Expected: {expected}\n"
                    "Either provide --checkpoint or pass --allow-hf-download after Hugging Face login."
                )

            checkpoint_info = {
                "source": "local" if checkpoint_path is not None else "huggingface_auto",
                "path": str(checkpoint_path) if checkpoint_path is not None else None,
            }

            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            load_start = time.perf_counter()
            model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                load_from_HF=checkpoint_path is None and args.allow_hf_download,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )
            processor = Sam3Processor(
                model,
                device=device,
                confidence_threshold=args.base_confidence,
            )

            if device == "cuda":
                torch.cuda.synchronize()

            model_load_seconds = time.perf_counter() - load_start
            print(f"  Model load time     : {model_load_seconds:.3f} s")

            use_autocast = (
                device == "cuda"
                and not args.no_autocast
                and bool(environment.get("bf16_supported", False))
            )

            inference_times: list[float] = []

            for index, row in enumerate(pending_rows, start=1):
                sample_id = row["sample_id"]
                image_path = resolve_row_path(project_root, row["processed_image_path"])
                mask_path = resolve_row_path(project_root, row["processed_mask_path"])

                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                with Image.open(image_path) as image_file:
                    image = image_file.convert("RGB")

                gt_mask, gt_ids = load_gt_mask(mask_path)
                width, height = image.size

                if gt_mask.shape != (height, width):
                    raise RuntimeError(
                        f"Image/GT size mismatch for {sample_id}: image={(width, height)}, gt={gt_mask.shape}"
                    )

                masks, boxes, scores, elapsed, autocast_label = run_sam3_inference_one(
                    processor,
                    image,
                    args.prompt,
                    device,
                    use_autocast,
                )

                predictions = build_prediction_records(masks, boxes, scores, args.prompt)
                signature = cache_signature(row, args.prompt, args.base_confidence)
                cache_path = prediction_cache_path(prediction_dir, sample_id)
                save_prediction_cache(
                    cache_path,
                    signature,
                    row,
                    (width, height),
                    len(gt_ids),
                    predictions,
                    elapsed,
                    autocast_label,
                )

                data = read_json(cache_path)
                cached_data_by_sample[sample_id] = data
                inference_times.append(elapsed)

                print(
                    f"  [{index:03d}/{len(pending_rows):03d}] "
                    f"{sample_id} | GT={len(gt_ids):3d} | pred={len(predictions):3d} | "
                    f"{elapsed:.3f}s"
                )

            if device == "cuda":
                peak_memory_bytes = int(torch.cuda.max_memory_allocated())

            summary["new_inference"] = {
                "images": len(pending_rows),
                "mean_seconds_per_image": float(np.mean(inference_times)) if inference_times else 0.0,
                "median_seconds_per_image": float(np.median(inference_times)) if inference_times else 0.0,
                "model_load_seconds": model_load_seconds,
                "cuda_peak_memory_bytes": peak_memory_bytes,
                "cuda_peak_memory": human_bytes(peak_memory_bytes),
            }
        else:
            print("  [SKIP] All selected images have compatible cached predictions.")
            summary["new_inference"] = {
                "images": 0,
                "mean_seconds_per_image": 0.0,
                "median_seconds_per_image": 0.0,
                "model_load_seconds": 0.0,
                "cuda_peak_memory_bytes": None,
                "cuda_peak_memory": None,
            }

        if package_info is not None:
            summary["sam3_package"] = package_info
        if checkpoint_info is not None:
            summary["checkpoint"] = checkpoint_info

        # Refresh cache index states after successful inference.
        refreshed_cache_index: list[dict] = []
        for row in selected_rows:
            cache_path = prediction_cache_path(prediction_dir, row["sample_id"])
            refreshed_cache_index.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row.get("split"),
                    "cache_state": "ready" if cache_path.is_file() else "missing",
                    "cache_path": relative_or_absolute(cache_path, project_root),
                }
            )
        write_csv(
            cache_index_csv,
            refreshed_cache_index,
            ["sample_id", "split", "cache_state", "cache_path"],
        )

        # ------------------------------------------------------------------
        # 4/6 Build evaluation records
        # ------------------------------------------------------------------
        print()
        print("[4/6] Building GT/prediction IoU matrices...")

        eval_records: list[dict] = []
        total_cached_predictions = 0
        total_gt_instances = 0

        for index, row in enumerate(selected_rows, start=1):
            sample_id = row["sample_id"]
            cache_path = prediction_cache_path(prediction_dir, sample_id)
            cache_data = read_json(cache_path)

            expected_signature = cache_signature(row, args.prompt, args.base_confidence)
            if not cache_is_compatible(cache_data, expected_signature):
                raise RuntimeError(f"Prediction cache is incompatible for sample {sample_id}: {cache_path}")

            record = load_eval_record(project_root, row, cache_data)
            eval_records.append(record)
            total_cached_predictions += record["pred_count"]
            total_gt_instances += record["gt_count"]

            if index % 20 == 0 or index == len(selected_rows):
                print(f"  Built IoU matrices: {index}/{len(selected_rows)}")

        summary["evaluation_dataset"] = {
            "images": len(eval_records),
            "ground_truth_instances": total_gt_instances,
            "cached_predictions_at_base_confidence": total_cached_predictions,
            "mean_cached_predictions_per_image": safe_divide(total_cached_predictions, len(eval_records)),
        }

        # ------------------------------------------------------------------
        # 5/6 Threshold sweep + AP
        # ------------------------------------------------------------------
        print()
        print("[5/6] Evaluating confidence thresholds and mask AP...")

        sweep_rows: list[dict] = []
        per_image_by_threshold: dict[float, list[dict]] = {}

        for threshold in thresholds:
            metrics, per_image_rows = evaluate_threshold(
                eval_records,
                threshold,
                args.match_iou,
                args.auto_accept_iou,
            )
            sweep_rows.append(metrics)
            per_image_by_threshold[threshold] = per_image_rows

            print(
                f"  conf={threshold:.2f} | "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} | "
                f"meanIoU={metrics['mean_matched_iou']:.4f} | "
                f"AutoAccept@{args.auto_accept_iou:.2f}={metrics['auto_accept_rate_gt']:.4f}"
            )

        # Highest F1; ties resolved by higher precision, then higher threshold.
        best_metrics = max(
            sweep_rows,
            key=lambda row: (row["f1"], row["precision"], row["confidence_threshold"]),
        )
        best_threshold = float(best_metrics["confidence_threshold"])

        ap50 = compute_average_precision(eval_records, 0.50)
        ap75 = compute_average_precision(eval_records, 0.75)

        summary["threshold_sweep"] = sweep_rows
        summary["best_validation_threshold_by_f1"] = best_threshold
        summary["best_validation_metrics"] = best_metrics
        summary["mask_ap"] = {
            "note": (
                "AP is computed from all predictions returned at the base SAM 3 confidence threshold; "
                "therefore predictions below base_confidence are unavailable by design."
            ),
            "base_confidence": args.base_confidence,
            "ap50": ap50,
            "ap75": ap75,
        }

        sweep_csv_rows: list[dict] = []
        for row in sweep_rows:
            sweep_csv_rows.append(
                {
                    "confidence_threshold": row["confidence_threshold"],
                    "images": row["images"],
                    "ground_truth_instances": row["ground_truth_instances"],
                    "kept_predictions": row["kept_predictions"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "precision": round(row["precision"], 6),
                    "recall": round(row["recall"], 6),
                    "f1": round(row["f1"], 6),
                    "mean_matched_iou": round(row["mean_matched_iou"], 6),
                    "auto_accept_tp": row["auto_accept_tp"],
                    "auto_accept_rate_gt": round(row["auto_accept_rate_gt"], 6),
                    "mean_predictions_per_image": round(row["mean_predictions_per_image"], 6),
                }
            )

        write_csv(
            sweep_csv,
            sweep_csv_rows,
            [
                "confidence_threshold",
                "images",
                "ground_truth_instances",
                "kept_predictions",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_matched_iou",
                "auto_accept_tp",
                "auto_accept_rate_gt",
                "mean_predictions_per_image",
            ],
        )

        best_per_image_rows = per_image_by_threshold[best_threshold]
        write_csv(
            per_image_csv,
            best_per_image_rows,
            [
                "sample_id",
                "confidence_threshold",
                "gt_instances",
                "cached_predictions",
                "kept_predictions",
                "tp_iou50",
                "fp_iou50",
                "fn_iou50",
                "precision_iou50",
                "recall_iou50",
                "f1_iou50",
                "mean_matched_iou",
                "auto_accept_tp",
                "auto_accept_rate_gt",
                "inference_seconds",
            ],
        )

        save_threshold_plot(plot_path, sweep_rows)

        # ------------------------------------------------------------------
        # 6/6 Final report
        # ------------------------------------------------------------------
        print()
        print("[6/6] Saving final baseline report...")

        all_inference_times = [record["inference_seconds"] for record in eval_records]
        summary["runtime"] = {
            "mean_inference_seconds_per_image": float(np.mean(all_inference_times))
            if all_inference_times
            else 0.0,
            "median_inference_seconds_per_image": float(np.median(all_inference_times))
            if all_inference_times
            else 0.0,
            "total_cached_inference_seconds": float(np.sum(all_inference_times)),
        }

        summary["outputs"] = {
            "experiment_root": relative_or_absolute(experiment_root, project_root),
            "prediction_directory": relative_or_absolute(prediction_dir, project_root),
            "prediction_cache_index_csv": relative_or_absolute(cache_index_csv, project_root),
            "threshold_sweep_csv": relative_or_absolute(sweep_csv, project_root),
            "best_threshold_per_image_csv": relative_or_absolute(per_image_csv, project_root),
            "threshold_plot": relative_or_absolute(plot_path, project_root)
            if plot_path.exists()
            else None,
            "summary_json": relative_or_absolute(summary_path, project_root),
        }

        summary["status"] = "PASS_WITH_WARNINGS" if summary["warnings"] else "PASS"
        write_json(summary_path, summary)

        print()
        print("=" * 88)
        print("SAM 3 ZERO-SHOT BASELINE COMPLETE")
        print("=" * 88)
        print(f"STATUS                        : {summary['status']}")
        print(f"SPLIT                         : {args.split}")
        print(f"IMAGES                        : {len(eval_records)}")
        print(f"GT APPLE INSTANCES            : {total_gt_instances}")
        print(f"BASE CONFIDENCE               : {args.base_confidence:.2f}")
        print(f"CACHED PREDICTIONS            : {total_cached_predictions}")
        print(f"BEST THRESHOLD BY F1          : {best_threshold:.2f}")
        print(f"PRECISION @IoU{args.match_iou:.2f}             : {best_metrics['precision']:.4f}")
        print(f"RECALL @IoU{args.match_iou:.2f}                : {best_metrics['recall']:.4f}")
        print(f"F1 @IoU{args.match_iou:.2f}                    : {best_metrics['f1']:.4f}")
        print(f"MEAN MATCHED MASK IoU         : {best_metrics['mean_matched_iou']:.4f}")
        print(
            f"AUTO ACCEPT RATE @IoU{args.auto_accept_iou:.2f}  : "
            f"{best_metrics['auto_accept_rate_gt']:.4f}"
        )
        print(f"MASK AP50                     : {ap50['ap']:.4f}")
        print(f"MASK AP75                     : {ap75['ap']:.4f}")
        print(
            f"MEAN INFERENCE TIME / IMAGE   : "
            f"{summary['runtime']['mean_inference_seconds_per_image']:.3f} s"
        )
        print(f"THRESHOLD SWEEP               : {sweep_csv}")
        print(f"PER-IMAGE METRICS             : {per_image_csv}")
        print(f"SUMMARY                       : {summary_path}")
        print("=" * 88)

    except Exception as error:
        summary["status"] = "FAIL"
        summary["errors"].append(str(error))
        summary["exception_type"] = type(error).__name__
        summary["traceback"] = traceback.format_exc()
        write_json(summary_path, summary)

        print()
        print("=" * 88)
        print("SAM 3 ZERO-SHOT BASELINE FAILED")
        print("=" * 88)
        print(f"ERROR   : {error}")
        print(f"REPORT  : {summary_path}")
        print("=" * 88)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
