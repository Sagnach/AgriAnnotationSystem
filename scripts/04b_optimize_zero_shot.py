from __future__ import annotations

from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from importlib import metadata as importlib_metadata

import argparse
import csv
import hashlib
import itertools
import json
import platform
import re
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
from PIL import Image


# =============================================================================
# SAM 3 Zero-Shot Optimization for AgriAnnotationSystem
# =============================================================================
#
# Purpose
# -------
# 1. Use ONLY the frozen MinneApple validation split for optimization by default.
# 2. Reuse the Stage-4 baseline cache whenever a compatible prediction exists.
# 3. Run additional SAM 3 text prompts only when their prediction cache is absent.
# 4. Stage A: compare prompt x confidence-threshold combinations.
# 5. Stage B: optimize lightweight post-processing on the best prompt(s):
#       - confidence threshold
#       - minimum mask area
#       - maximum mask-area fraction
#       - score-ordered mask-IoU duplicate suppression (mask NMS)
# 6. Report standard metrics and annotation-oriented metrics:
#       - Precision / Recall / F1 / F2 @ match IoU
#       - mean matched mask IoU
#       - AutoAccept rates @ configurable IoUs
#       - Delete / Refine / Add action proxy for human review
# 7. Compute AP50 / AP75 after the selected STRUCTURAL post-processing filters.
# 8. Save all optimization tables, selected per-image metrics and summary JSON.
#
# IMPORTANT
# ---------
# The official MinneApple test split is frozen. This optimization script refuses
# to use split=test unless --allow-test is supplied explicitly. For the intended
# workflow, optimize on val only, freeze the configuration, then use a separate
# final-test script/run.
#
# This script does NOT train SAM 3.
#
# =============================================================================


MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)
MIN_CUDA = (12, 6)

DEFAULT_PROMPTS = [
    "apple",
    "apples",
    "apple fruit",
    "apples on tree",
    "orchard apple",
]

DEFAULT_BASE_CONFIDENCE = 0.25
DEFAULT_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
DEFAULT_MIN_AREA_PIXELS = [0, 25, 50, 100, 200]
DEFAULT_MAX_AREA_FRACTIONS = [1.0, 0.05, 0.02, 0.01]
# 0.0 means mask-NMS disabled.
DEFAULT_NMS_IOUS = [0.0, 0.90, 0.80, 0.70]
DEFAULT_MATCH_IOU = 0.50
DEFAULT_ACCEPT_IOUS = [0.70, 0.75, 0.80, 0.85]
DEFAULT_TOP_PROMPTS = 2
DEFAULT_OBJECTIVE = "f1"

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

    # Expected location:
    # AgriAnnotationSystem/scripts/04b_optimize_zero_shot.py
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


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def fbeta_score(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denominator)


def prompt_slug(prompt: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower()).strip("_")
    cleaned = cleaned[:40] if cleaned else "prompt"
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}__{digest}"


def parse_unique_floats(
    values: list[float] | None,
    default: list[float],
    name: str,
    minimum: float,
    maximum: float,
    allow_zero: bool = True,
) -> list[float]:
    raw = values if values is not None and len(values) > 0 else default
    parsed = sorted(set(round(float(v), 8) for v in raw))

    if not parsed:
        raise ValueError(f"{name}: at least one value is required.")

    for value in parsed:
        if value < minimum or value > maximum:
            raise ValueError(f"{name}: invalid value {value}; expected [{minimum}, {maximum}].")
        if not allow_zero and value == 0:
            raise ValueError(f"{name}: zero is not allowed.")

    return parsed


def parse_unique_ints(
    values: list[int] | None,
    default: list[int],
    name: str,
    minimum: int = 0,
) -> list[int]:
    raw = values if values is not None and len(values) > 0 else default
    parsed = sorted(set(int(v) for v in raw))

    if not parsed:
        raise ValueError(f"{name}: at least one value is required.")

    for value in parsed:
        if value < minimum:
            raise ValueError(f"{name}: invalid value {value}; expected >= {minimum}.")

    return parsed


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
            "SAM 3 could not be imported. Ensure the official repository and "
            "inference dependencies are installed in the active environment."
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
            path
            for path in candidates
            if "sam3" in path.name.lower() and "3.1" not in path.name.lower()
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

    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")

    required = {
        "sample_id",
        "split",
        "instance_count",
        "processed_image_path",
        "processed_mask_path",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {sorted(missing)}")

    return rows


def select_split_rows(
    rows: list[dict[str, str]],
    split: str,
    limit: int | None,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("split", "").strip().lower() == split.strip().lower()
    ]
    selected = sorted(selected, key=lambda row: row.get("sample_id", ""))

    if not selected:
        raise RuntimeError(f"No rows found for split={split!r}.")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        selected = selected[:limit]

    return selected


def load_gt_indexed_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {path}")

    with Image.open(path) as image:
        mask = np.asarray(image)

    if mask.ndim != 2:
        raise RuntimeError(f"Expected a 2-D indexed GT mask; found {mask.shape}")

    ids = np.unique(mask)
    ids = ids[ids != 0]
    return mask, ids


# =============================================================================
# Tensor / SAM prediction conversion
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
# COCO RLE helpers
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


def coco_rle(rle: dict) -> dict:
    """Convert a JSON-safe RLE back to the pycocotools byte-count form."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")

    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": counts,
    }


def gt_rles_from_indexed(mask: np.ndarray, ids: np.ndarray) -> list[dict]:
    return [encode_binary_mask(mask == instance_id) for instance_id in ids]


# =============================================================================
# Prediction cache
# =============================================================================


def cache_signature(sample: dict[str, str], prompt: str, base_confidence: float) -> dict:
    return {
        "sample_id": sample.get("sample_id"),
        "image_path": sample.get("processed_image_path"),
        "mask_path": sample.get("processed_mask_path"),
        "prompt": prompt,
        "base_confidence": round(float(base_confidence), 6),
    }


def cache_is_compatible(data: dict, expected_signature: dict) -> bool:
    return data.get("signature") == expected_signature and data.get("status") == "PASS"


def optimizer_cache_path(root: Path, prompt: str, sample_id: str) -> Path:
    return root / "predictions" / prompt_slug(prompt) / f"{sample_id}.json"


def baseline_cache_path(baseline_prediction_dir: Path, sample_id: str) -> Path:
    return baseline_prediction_dir / f"{sample_id}.json"


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

    boxes = (
        np.empty((0, 4), dtype=np.float32)
        if boxes.size == 0
        else np.asarray(boxes).reshape(-1, 4)
    )

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


def find_compatible_cache(
    sample: dict[str, str],
    prompt: str,
    base_confidence: float,
    optimization_root: Path,
    baseline_prediction_dir: Path | None,
    reuse_baseline: bool,
) -> tuple[Path | None, str]:
    expected = cache_signature(sample, prompt, base_confidence)

    own_path = optimizer_cache_path(optimization_root, prompt, sample["sample_id"])
    if own_path.is_file():
        try:
            data = read_json(own_path)
            if cache_is_compatible(data, expected):
                return own_path, "optimizer_cache"
        except Exception:
            pass

    if reuse_baseline and baseline_prediction_dir is not None:
        candidate = baseline_cache_path(baseline_prediction_dir, sample["sample_id"])
        if candidate.is_file():
            try:
                data = read_json(candidate)
                if cache_is_compatible(data, expected):
                    return candidate, "baseline_reused"
            except Exception:
                pass

    return None, "missing"


# =============================================================================
# IoU matrices via pycocotools
# =============================================================================


def rle_iou_matrix(dt_rles: list[dict], gt_rles: list[dict]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    num_dt = len(dt_rles)
    num_gt = len(gt_rles)
    if num_dt == 0 or num_gt == 0:
        return np.zeros((num_dt, num_gt), dtype=np.float32)

    dt = [coco_rle(rle) for rle in dt_rles]
    gt = [coco_rle(rle) for rle in gt_rles]
    matrix = mask_utils.iou(dt, gt, [0] * len(gt))
    return np.asarray(matrix, dtype=np.float32)


def pred_pairwise_iou_matrix(pred_rles: list[dict]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    count = len(pred_rles)
    if count == 0:
        return np.zeros((0, 0), dtype=np.float32)

    rles = [coco_rle(rle) for rle in pred_rles]
    matrix = mask_utils.iou(rles, rles, [0] * len(rles))
    return np.asarray(matrix, dtype=np.float32)


def load_eval_record(
    project_root: Path,
    sample: dict[str, str],
    cache_data: dict,
    prompt: str,
) -> dict:
    mask_path = resolve_row_path(project_root, sample["processed_mask_path"])
    gt_mask, gt_ids = load_gt_indexed_mask(mask_path)
    gt_rles = gt_rles_from_indexed(gt_mask, gt_ids)

    predictions = cache_data.get("predictions", [])
    scores = np.asarray([float(item["score"]) for item in predictions], dtype=np.float32)
    areas = np.asarray(
        [int(item.get("mask_area_pixels", 0)) for item in predictions],
        dtype=np.int64,
    )
    pred_rles = [item["rle"] for item in predictions]

    gt_iou = rle_iou_matrix(pred_rles, gt_rles)
    pred_iou = pred_pairwise_iou_matrix(pred_rles)

    height, width = gt_mask.shape

    return {
        "sample_id": sample["sample_id"],
        "prompt": prompt,
        "gt_count": int(len(gt_ids)),
        "pred_count": int(len(predictions)),
        "scores": scores,
        "areas": areas,
        "image_area": int(height * width),
        "gt_iou_matrix": gt_iou,
        "pred_iou_matrix": pred_iou,
        "inference_seconds": float(cache_data.get("inference", {}).get("seconds", 0.0)),
    }


# =============================================================================
# Post-processing and matching
# =============================================================================


def select_prediction_indices(
    record: dict,
    confidence_threshold: float,
    min_area_pixels: int,
    max_area_fraction: float,
    nms_iou: float,
) -> np.ndarray:
    scores = record["scores"]
    areas = record["areas"]
    max_area_pixels = float(record["image_area"]) * float(max_area_fraction)

    keep = np.where(
        (scores >= confidence_threshold)
        & (areas >= int(min_area_pixels))
        & (areas <= max_area_pixels)
    )[0]

    if len(keep) == 0:
        return keep.astype(np.int64)

    order = keep[np.argsort(scores[keep])[::-1]]

    # nms_iou <= 0 explicitly means "off".
    if nms_iou <= 0.0:
        return order.astype(np.int64)

    pairwise = record["pred_iou_matrix"]
    selected: list[int] = []

    for pred_index in order:
        pred_index = int(pred_index)
        suppress = False

        for kept_index in selected:
            if float(pairwise[pred_index, kept_index]) >= nms_iou:
                suppress = True
                break

        if not suppress:
            selected.append(pred_index)

    return np.asarray(selected, dtype=np.int64)


def match_selected_predictions(
    gt_iou_matrix: np.ndarray,
    selected_indices: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> dict:
    num_gt = int(gt_iou_matrix.shape[1])

    if len(selected_indices):
        ordered = selected_indices[np.argsort(scores[selected_indices])[::-1]]
    else:
        ordered = selected_indices

    matched_gt: set[int] = set()
    matches: list[dict] = []
    tp = 0
    fp = 0

    for pred_index in ordered:
        pred_index = int(pred_index)

        if num_gt == 0:
            fp += 1
            matches.append(
                {
                    "pred_index": pred_index,
                    "gt_index": None,
                    "iou": 0.0,
                    "score": float(scores[pred_index]),
                    "is_tp": False,
                }
            )
            continue

        row = gt_iou_matrix[pred_index]
        gt_order = np.argsort(row)[::-1]
        chosen_gt = None
        chosen_iou = 0.0

        for gt_index in gt_order:
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
                "pred_index": pred_index,
                "gt_index": int(chosen_gt) if is_tp and chosen_gt is not None else None,
                "iou": chosen_iou,
                "score": float(scores[pred_index]),
                "is_tp": is_tp,
            }
        )

    fn = num_gt - tp
    matched_ious = [item["iou"] for item in matches if item["is_tp"]]

    return {
        "selected_predictions": int(len(ordered)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "matches": matches,
        "matched_ious": matched_ious,
    }


def evaluate_config(
    records: list[dict],
    confidence_threshold: float,
    min_area_pixels: int,
    max_area_fraction: float,
    nms_iou: float,
    match_iou: float,
    accept_ious: list[float],
    include_per_image: bool = False,
) -> tuple[dict, list[dict]]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_selected = 0
    all_matched_ious: list[float] = []
    accept_counts = {value: 0 for value in accept_ious}
    per_image_rows: list[dict] = []

    strict_accept = max(accept_ious)

    for record in records:
        selected = select_prediction_indices(
            record,
            confidence_threshold,
            min_area_pixels,
            max_area_fraction,
            nms_iou,
        )

        matched = match_selected_predictions(
            record["gt_iou_matrix"],
            selected,
            record["scores"],
            match_iou,
        )

        matched_ious = matched["matched_ious"]

        total_tp += matched["tp"]
        total_fp += matched["fp"]
        total_fn += matched["fn"]
        total_gt += record["gt_count"]
        total_selected += matched["selected_predictions"]
        all_matched_ious.extend(matched_ious)

        local_accept_counts = {}
        for threshold in accept_ious:
            value = sum(1 for iou in matched_ious if iou >= threshold)
            accept_counts[threshold] += value
            local_accept_counts[threshold] = value

        precision = safe_divide(matched["tp"], matched["tp"] + matched["fp"])
        recall = safe_divide(matched["tp"], matched["tp"] + matched["fn"])
        f1 = fbeta_score(precision, recall, 1.0)
        f2 = fbeta_score(precision, recall, 2.0)

        strict_auto = local_accept_counts[strict_accept]
        refine_actions = max(0, matched["tp"] - strict_auto)
        review_actions = matched["fp"] + matched["fn"] + refine_actions

        if include_per_image:
            row = {
                "sample_id": record["sample_id"],
                "prompt": record["prompt"],
                "confidence_threshold": confidence_threshold,
                "min_area_pixels": min_area_pixels,
                "max_area_fraction": max_area_fraction,
                "mask_nms_iou": nms_iou,
                "gt_instances": record["gt_count"],
                "cached_predictions": record["pred_count"],
                "kept_predictions": matched["selected_predictions"],
                "tp": matched["tp"],
                "fp_delete": matched["fp"],
                "fn_add": matched["fn"],
                "refine_actions": refine_actions,
                "review_actions_proxy": review_actions,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "f2": round(f2, 6),
                "mean_matched_iou": round(float(np.mean(matched_ious)), 6)
                if matched_ious
                else 0.0,
            }
            for threshold in accept_ious:
                key = f"auto_accept_{str(threshold).replace('.', '_')}"
                row[key] = local_accept_counts[threshold]
            per_image_rows.append(row)

    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)
    f1 = fbeta_score(precision, recall, 1.0)
    f2 = fbeta_score(precision, recall, 2.0)
    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    strict_auto = accept_counts[strict_accept]
    refine_actions = max(0, total_tp - strict_auto)
    review_actions = total_fp + total_fn + refine_actions

    metrics = {
        "prompt": records[0]["prompt"] if records else "",
        "confidence_threshold": float(confidence_threshold),
        "min_area_pixels": int(min_area_pixels),
        "max_area_fraction": float(max_area_fraction),
        "mask_nms_iou": float(nms_iou),
        "matching_iou_threshold": float(match_iou),
        "images": len(records),
        "ground_truth_instances": int(total_gt),
        "kept_predictions": int(total_selected),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "mean_matched_iou": mean_iou,
        "mean_predictions_per_image": safe_divide(total_selected, len(records)),
        "delete_actions_proxy": int(total_fp),
        "add_actions_proxy": int(total_fn),
        "refine_actions_proxy": int(refine_actions),
        "review_actions_proxy": int(review_actions),
        "review_actions_per_image": safe_divide(review_actions, len(records)),
        "review_actions_per_gt": safe_divide(review_actions, total_gt),
    }

    for threshold in accept_ious:
        label = str(threshold).replace(".", "_")
        count = accept_counts[threshold]
        metrics[f"auto_accept_tp_{label}"] = int(count)
        metrics[f"auto_accept_rate_gt_{label}"] = safe_divide(count, total_gt)

    return metrics, per_image_rows


# =============================================================================
# Ranking / AP
# =============================================================================


def config_rank_key(row: dict, objective: str):
    if objective == "f1":
        return (
            row["f1"],
            row["f2"],
            row["recall"],
            row["precision"],
            -row["review_actions_proxy"],
        )

    if objective == "f2":
        return (
            row["f2"],
            row["recall"],
            row["f1"],
            row["precision"],
            -row["review_actions_proxy"],
        )

    if objective == "review_actions":
        # max() is used, so negate lower-is-better quantities.
        return (
            -row["review_actions_proxy"],
            -row["fn"],
            row["f2"],
            row["f1"],
            row["recall"],
        )

    raise ValueError(f"Unknown objective: {objective}")


def best_row(rows: list[dict], objective: str) -> dict:
    if not rows:
        raise RuntimeError("No optimization rows were generated.")
    return max(rows, key=lambda row: config_rank_key(row, objective))


def compute_filtered_average_precision(
    records: list[dict],
    min_area_pixels: int,
    max_area_fraction: float,
    nms_iou: float,
    base_confidence: float,
    iou_threshold: float,
) -> dict:
    total_gt = sum(record["gt_count"] for record in records)

    candidates: list[tuple[float, int, int]] = []

    for image_index, record in enumerate(records):
        selected = select_prediction_indices(
            record,
            confidence_threshold=base_confidence,
            min_area_pixels=min_area_pixels,
            max_area_fraction=max_area_fraction,
            nms_iou=nms_iou,
        )

        for pred_index in selected:
            candidates.append(
                (float(record["scores"][pred_index]), image_index, int(pred_index))
            )

    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_gt_by_image: dict[int, set[int]] = defaultdict(set)
    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for _, image_index, pred_index in candidates:
        record = records[image_index]
        row = record["gt_iou_matrix"][pred_index]
        matched_gt = matched_gt_by_image[image_index]

        if len(row) == 0:
            tp_flags.append(0)
            fp_flags.append(1)
            continue

        order = np.argsort(row)[::-1]
        chosen_gt = None
        chosen_iou = 0.0

        for gt_index in order:
            gt_index = int(gt_index)
            if gt_index in matched_gt:
                continue
            chosen_gt = gt_index
            chosen_iou = float(row[gt_index])
            break

        if chosen_gt is not None and chosen_iou >= iou_threshold:
            matched_gt.add(chosen_gt)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    if total_gt == 0 or not candidates:
        return {
            "iou_threshold": float(iou_threshold),
            "ap": 0.0,
            "total_gt": int(total_gt),
            "total_predictions": int(len(candidates)),
            "max_recall": 0.0,
            "final_precision": 0.0,
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

    return {
        "iou_threshold": float(iou_threshold),
        "ap": float(np.mean(interpolated)),
        "total_gt": int(total_gt),
        "total_predictions": int(len(candidates)),
        "max_recall": float(recall[-1]) if len(recall) else 0.0,
        "final_precision": float(precision[-1]) if len(precision) else 0.0,
    }


# =============================================================================
# Plot helpers
# =============================================================================


def save_prompt_sweep_plot(path: Path, rows: list[dict], metric: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    prompts = sorted(set(row["prompt"] for row in rows))

    fig, ax = plt.subplots(figsize=(10, 6))
    for prompt in prompts:
        subset = sorted(
            [row for row in rows if row["prompt"] == prompt],
            key=lambda row: row["confidence_threshold"],
        )
        ax.plot(
            [row["confidence_threshold"] for row in subset],
            [row[metric] for row in subset],
            marker="o",
            label=prompt,
        )

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# CSV formatting
# =============================================================================


def flatten_metric_row(row: dict, accept_ious: list[float]) -> dict:
    output = {
        "prompt": row["prompt"],
        "confidence_threshold": row["confidence_threshold"],
        "min_area_pixels": row["min_area_pixels"],
        "max_area_fraction": row["max_area_fraction"],
        "mask_nms_iou": row["mask_nms_iou"],
        "images": row["images"],
        "ground_truth_instances": row["ground_truth_instances"],
        "kept_predictions": row["kept_predictions"],
        "tp": row["tp"],
        "fp": row["fp"],
        "fn": row["fn"],
        "precision": round(row["precision"], 6),
        "recall": round(row["recall"], 6),
        "f1": round(row["f1"], 6),
        "f2": round(row["f2"], 6),
        "mean_matched_iou": round(row["mean_matched_iou"], 6),
        "mean_predictions_per_image": round(row["mean_predictions_per_image"], 6),
        "delete_actions_proxy": row["delete_actions_proxy"],
        "add_actions_proxy": row["add_actions_proxy"],
        "refine_actions_proxy": row["refine_actions_proxy"],
        "review_actions_proxy": row["review_actions_proxy"],
        "review_actions_per_image": round(row["review_actions_per_image"], 6),
        "review_actions_per_gt": round(row["review_actions_per_gt"], 6),
    }

    for threshold in accept_ious:
        label = str(threshold).replace(".", "_")
        output[f"auto_accept_tp_{label}"] = row[f"auto_accept_tp_{label}"]
        output[f"auto_accept_rate_gt_{label}"] = round(
            row[f"auto_accept_rate_gt_{label}"], 6
        )

    return output


def metric_fieldnames(accept_ious: list[float]) -> list[str]:
    fields = [
        "prompt",
        "confidence_threshold",
        "min_area_pixels",
        "max_area_fraction",
        "mask_nms_iou",
        "images",
        "ground_truth_instances",
        "kept_predictions",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "f2",
        "mean_matched_iou",
        "mean_predictions_per_image",
        "delete_actions_proxy",
        "add_actions_proxy",
        "refine_actions_proxy",
        "review_actions_proxy",
        "review_actions_per_image",
        "review_actions_per_gt",
    ]

    for threshold in accept_ious:
        label = str(threshold).replace(".", "_")
        fields.extend(
            [
                f"auto_accept_tp_{label}",
                f"auto_accept_rate_gt_{label}",
            ]
        )

    return fields


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the SAM 3 zero-shot MinneApple validation pipeline with "
            "prompt selection and lightweight post-processing."
        )
    )

    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--base-confidence", type=float, default=DEFAULT_BASE_CONFIDENCE)
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--min-area-pixels", nargs="*", type=int, default=None)
    parser.add_argument("--max-area-fractions", nargs="*", type=float, default=None)
    parser.add_argument("--nms-ious", nargs="*", type=float, default=None)
    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--accept-ious", nargs="*", type=float, default=None)
    parser.add_argument("--top-prompts", type=int, default=DEFAULT_TOP_PROMPTS)
    parser.add_argument(
        "--objective",
        choices=["f1", "f2", "review_actions"],
        default=DEFAULT_OBJECTIVE,
        help=(
            "How to select the final validation configuration. f1 is the default "
            "benchmark objective; f2 favors recall; review_actions minimizes the "
            "Delete+Refine+Add proxy count."
        ),
    )

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-reuse-baseline-cache",
        action="store_true",
        help="Do not reuse compatible Stage-4 caches from sam3_zero_shot_<split>/predictions.",
    )

    args = parser.parse_args()

    prompts = args.prompts if args.prompts else list(DEFAULT_PROMPTS)
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        raise SystemExit("At least one prompt is required.")

    thresholds = parse_unique_floats(
        args.thresholds,
        DEFAULT_THRESHOLDS,
        "--thresholds",
        0.0,
        1.0,
    )
    min_area_pixels = parse_unique_ints(
        args.min_area_pixels,
        DEFAULT_MIN_AREA_PIXELS,
        "--min-area-pixels",
        minimum=0,
    )
    max_area_fractions = parse_unique_floats(
        args.max_area_fractions,
        DEFAULT_MAX_AREA_FRACTIONS,
        "--max-area-fractions",
        0.0,
        1.0,
        allow_zero=False,
    )
    nms_ious = parse_unique_floats(
        args.nms_ious,
        DEFAULT_NMS_IOUS,
        "--nms-ious",
        0.0,
        1.0,
    )
    accept_ious = parse_unique_floats(
        args.accept_ious,
        DEFAULT_ACCEPT_IOUS,
        "--accept-ious",
        args.match_iou,
        1.0,
    )

    if not 0.0 <= args.base_confidence <= 1.0:
        raise SystemExit("--base-confidence must be between 0 and 1.")
    if min(thresholds) < args.base_confidence:
        raise SystemExit(
            "The smallest optimization confidence threshold cannot be below "
            "--base-confidence because lower-score predictions are not cached."
        )
    if not 0.0 <= args.match_iou <= 1.0:
        raise SystemExit("--match-iou must be between 0 and 1.")
    if args.top_prompts <= 0:
        raise SystemExit("--top-prompts must be > 0.")
    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "The official test split is frozen. Optimize on validation only. "
            "Pass --allow-test only if you explicitly intend to break that protocol."
        )

    project_root = resolve_project_root(args.project_root)
    manifest_path = resolve_manifest_path(project_root, args.manifest)

    experiment_name = f"sam3_zero_shot_optimization_{args.split}"
    if args.limit is not None:
        experiment_name += f"_limit{args.limit}"

    optimization_root = (project_root / OUTPUTS_RELATIVE_PATH / experiment_name).resolve()
    optimization_root.mkdir(parents=True, exist_ok=True)

    baseline_root = (
        project_root / OUTPUTS_RELATIVE_PATH / f"sam3_zero_shot_{args.split}"
    ).resolve()
    baseline_prediction_dir = baseline_root / "predictions"
    baseline_summary_path = baseline_root / "baseline_summary.json"

    summary_path = optimization_root / "optimization_summary.json"
    cache_index_csv = optimization_root / "cache_index.csv"
    prompt_sweep_csv = optimization_root / "prompt_sweep.csv"
    postprocess_sweep_csv = optimization_root / "postprocess_sweep.csv"
    selected_per_image_csv = optimization_root / "selected_per_image_metrics.csv"
    prompt_f1_plot = optimization_root / "prompt_sweep_f1.png"
    prompt_f2_plot = optimization_root / "prompt_sweep_f2.png"

    summary: dict = {
        "optimization_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "status": "RUNNING",
        "project_root": str(project_root),
        "manifest": relative_or_absolute(manifest_path, project_root),
        "split": args.split,
        "prompts": prompts,
        "base_confidence": args.base_confidence,
        "thresholds": thresholds,
        "min_area_pixels": min_area_pixels,
        "max_area_fractions": max_area_fractions,
        "mask_nms_ious": nms_ious,
        "matching_iou": args.match_iou,
        "auto_accept_ious": accept_ious,
        "top_prompts_for_postprocess": args.top_prompts,
        "selection_objective": args.objective,
        "errors": [],
        "warnings": [],
    }

    print()
    print("=" * 96)
    print("SAM 3 ZERO-SHOT OPTIMIZATION — MINNEAPPLE")
    print("=" * 96)
    print(f"Project root          : {project_root}")
    print(f"Manifest              : {manifest_path}")
    print(f"Split                 : {args.split}")
    print(f"Prompts               : {prompts}")
    print(f"Base confidence       : {args.base_confidence:.2f}")
    print(f"Thresholds            : {thresholds}")
    print(f"Min mask area pixels  : {min_area_pixels}")
    print(f"Max mask area frac    : {max_area_fractions}")
    print(f"Mask NMS IoUs         : {nms_ious}  (0 = off)")
    print(f"Matching IoU          : {args.match_iou:.2f}")
    print(f"AutoAccept IoUs       : {accept_ious}")
    print(f"Selection objective   : {args.objective}")
    print(f"Output                : {optimization_root}")
    print()

    try:
        # ------------------------------------------------------------------
        # 1/7 Environment + frozen validation data
        # ------------------------------------------------------------------
        print("[1/7] Auditing environment and loading frozen data...")

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
        total_manifest_gt = int(
            sum(int(float(row.get("instance_count", "0") or 0)) for row in selected_rows)
        )

        summary["selected_images"] = len(selected_rows)
        summary["selected_gt_instances_manifest"] = total_manifest_gt

        print(f"  Device               : {device}")
        print(f"  Selected images      : {len(selected_rows)}")
        print(f"  Manifest GT apples   : {total_manifest_gt}")

        # ------------------------------------------------------------------
        # 2/7 Inspect prediction caches for all prompts
        # ------------------------------------------------------------------
        print()
        print("[2/7] Inspecting prompt prediction caches...")

        cache_lookup: dict[tuple[str, str], Path] = {}
        pending: list[tuple[str, dict[str, str]]] = []
        cache_index_rows: list[dict] = []
        reuse_baseline = not args.no_reuse_baseline_cache

        for prompt in prompts:
            reused = 0
            missing = 0

            for row in selected_rows:
                sample_id = row["sample_id"]

                cache_path = None
                source = "missing"

                if not args.overwrite:
                    cache_path, source = find_compatible_cache(
                        row,
                        prompt,
                        args.base_confidence,
                        optimization_root,
                        baseline_prediction_dir if baseline_prediction_dir.is_dir() else None,
                        reuse_baseline,
                    )

                if cache_path is not None:
                    cache_lookup[(prompt, sample_id)] = cache_path
                    reused += 1
                else:
                    pending.append((prompt, row))
                    missing += 1

                cache_index_rows.append(
                    {
                        "prompt": prompt,
                        "prompt_slug": prompt_slug(prompt),
                        "sample_id": sample_id,
                        "cache_state": source if cache_path is not None else "pending",
                        "cache_path": (
                            relative_or_absolute(cache_path, project_root)
                            if cache_path is not None
                            else relative_or_absolute(
                                optimizer_cache_path(optimization_root, prompt, sample_id),
                                project_root,
                            )
                        ),
                    }
                )

            print(f"  {prompt!r:<24} reused={reused:3d} pending={missing:3d}")

        # ------------------------------------------------------------------
        # 3/7 Run missing prompt inference
        # ------------------------------------------------------------------
        print()
        print("[3/7] Running missing SAM 3 prompt inference...")

        model_load_seconds = 0.0
        peak_memory_bytes = None
        package_info = None
        checkpoint_info = None
        new_inference_times: list[float] = []

        if pending:
            build_sam3_image_model, Sam3Processor, package_info = import_sam3(project_root)
            checkpoint_path = resolve_checkpoint(project_root, args.checkpoint)

            if checkpoint_path is None and not args.allow_hf_download:
                expected = project_root / DEFAULT_CHECKPOINT_RELATIVE_PATH
                raise RuntimeError(
                    "No local SAM 3 checkpoint was found.\n"
                    f"Expected: {expected}\n"
                    "Provide --checkpoint or use --allow-hf-download after Hugging Face login."
                )

            checkpoint_info = {
                "source": "local" if checkpoint_path is not None else "huggingface_auto",
                "path": str(checkpoint_path) if checkpoint_path is not None else None,
            }

            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
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

            model_load_seconds = time.perf_counter() - start
            print(f"  Model load time      : {model_load_seconds:.3f} s")

            use_autocast = (
                device == "cuda"
                and not args.no_autocast
                and bool(environment.get("bf16_supported", False))
            )

            total_pending = len(pending)

            for index, (prompt, row) in enumerate(pending, start=1):
                sample_id = row["sample_id"]
                image_path = resolve_row_path(project_root, row["processed_image_path"])
                mask_path = resolve_row_path(project_root, row["processed_mask_path"])

                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                with Image.open(image_path) as image_file:
                    image = image_file.convert("RGB")

                gt_mask, gt_ids = load_gt_indexed_mask(mask_path)
                width, height = image.size
                if gt_mask.shape != (height, width):
                    raise RuntimeError(
                        f"Image/GT size mismatch for {sample_id}: "
                        f"image={(width, height)}, gt={gt_mask.shape}"
                    )

                masks, boxes, scores, seconds, autocast_label = run_sam3_inference_one(
                    processor,
                    image,
                    prompt,
                    device,
                    use_autocast,
                )

                predictions = build_prediction_records(masks, boxes, scores, prompt)
                output_path = optimizer_cache_path(optimization_root, prompt, sample_id)
                signature = cache_signature(row, prompt, args.base_confidence)

                save_prediction_cache(
                    output_path,
                    signature,
                    row,
                    image.size,
                    len(gt_ids),
                    predictions,
                    seconds,
                    autocast_label,
                )

                cache_lookup[(prompt, sample_id)] = output_path
                new_inference_times.append(seconds)

                if index % 20 == 0 or index == total_pending:
                    print(
                        f"  Inference progress    : {index}/{total_pending} | "
                        f"last prompt={prompt!r} | preds={len(predictions)} | {seconds:.3f}s"
                    )

            if device == "cuda":
                peak_memory_bytes = int(torch.cuda.max_memory_allocated())

            del processor
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        else:
            print("  All required predictions are already cached.")

        summary["sam3_package"] = package_info
        summary["checkpoint"] = checkpoint_info
        summary["new_inference"] = {
            "prediction_jobs": len(new_inference_times),
            "mean_seconds_per_job": float(np.mean(new_inference_times))
            if new_inference_times
            else 0.0,
            "median_seconds_per_job": float(np.median(new_inference_times))
            if new_inference_times
            else 0.0,
            "model_load_seconds": model_load_seconds,
            "cuda_peak_memory_bytes": peak_memory_bytes,
            "cuda_peak_memory": human_bytes(peak_memory_bytes),
        }

        # Refresh cache index after inference.
        refreshed_cache_index: list[dict] = []
        for prompt in prompts:
            for row in selected_rows:
                sample_id = row["sample_id"]
                path, source = find_compatible_cache(
                    row,
                    prompt,
                    args.base_confidence,
                    optimization_root,
                    baseline_prediction_dir if baseline_prediction_dir.is_dir() else None,
                    reuse_baseline,
                )
                if path is None:
                    raise RuntimeError(
                        f"No compatible cache exists after inference: prompt={prompt!r}, "
                        f"sample={sample_id}"
                    )
                cache_lookup[(prompt, sample_id)] = path
                refreshed_cache_index.append(
                    {
                        "prompt": prompt,
                        "prompt_slug": prompt_slug(prompt),
                        "sample_id": sample_id,
                        "cache_state": source,
                        "cache_path": relative_or_absolute(path, project_root),
                    }
                )

        write_csv(
            cache_index_csv,
            refreshed_cache_index,
            ["prompt", "prompt_slug", "sample_id", "cache_state", "cache_path"],
        )

        # ------------------------------------------------------------------
        # 4/7 Build compact IoU records per prompt
        # ------------------------------------------------------------------
        print()
        print("[4/7] Building GT-IoU and prediction-IoU matrices...")

        records_by_prompt: dict[str, list[dict]] = {}

        for prompt in prompts:
            prompt_records: list[dict] = []
            total_predictions = 0

            for index, row in enumerate(selected_rows, start=1):
                sample_id = row["sample_id"]
                path = cache_lookup[(prompt, sample_id)]
                cache_data = read_json(path)
                expected = cache_signature(row, prompt, args.base_confidence)

                if not cache_is_compatible(cache_data, expected):
                    raise RuntimeError(
                        f"Incompatible prediction cache: prompt={prompt!r}, sample={sample_id}"
                    )

                record = load_eval_record(project_root, row, cache_data, prompt)
                prompt_records.append(record)
                total_predictions += record["pred_count"]

                if index % 40 == 0 or index == len(selected_rows):
                    print(
                        f"  {prompt!r:<24} IoU matrices {index}/{len(selected_rows)}"
                    )

            records_by_prompt[prompt] = prompt_records
            print(
                f"    -> {prompt!r}: cached predictions={total_predictions}, "
                f"mean/image={safe_divide(total_predictions, len(prompt_records)):.2f}"
            )

        # ------------------------------------------------------------------
        # 5/7 Stage A — prompt + confidence optimization
        # ------------------------------------------------------------------
        print()
        print("[5/7] Stage A: prompt + confidence sweep...")

        prompt_sweep_rows: list[dict] = []

        for prompt in prompts:
            records = records_by_prompt[prompt]
            for threshold in thresholds:
                metrics, _ = evaluate_config(
                    records,
                    confidence_threshold=threshold,
                    min_area_pixels=0,
                    max_area_fraction=1.0,
                    nms_iou=0.0,
                    match_iou=args.match_iou,
                    accept_ious=accept_ious,
                    include_per_image=False,
                )
                prompt_sweep_rows.append(metrics)

                print(
                    f"  prompt={prompt!r:<20} conf={threshold:.2f} | "
                    f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                    f"F1={metrics['f1']:.4f} F2={metrics['f2']:.4f} | "
                    f"actions={metrics['review_actions_proxy']}"
                )

        prompt_best_rows: list[dict] = []
        for prompt in prompts:
            subset = [row for row in prompt_sweep_rows if row["prompt"] == prompt]
            prompt_best_rows.append(best_row(subset, args.objective))

        prompt_best_rows = sorted(
            prompt_best_rows,
            key=lambda row: config_rank_key(row, args.objective),
            reverse=True,
        )

        top_count = min(args.top_prompts, len(prompt_best_rows))
        top_prompts = [row["prompt"] for row in prompt_best_rows[:top_count]]

        print()
        print(f"  Top prompts for Stage B ({args.objective}) : {top_prompts}")

        write_csv(
            prompt_sweep_csv,
            [flatten_metric_row(row, accept_ious) for row in prompt_sweep_rows],
            metric_fieldnames(accept_ious),
        )

        save_prompt_sweep_plot(
            prompt_f1_plot,
            prompt_sweep_rows,
            metric="f1",
            title="SAM 3 MinneApple Validation — Prompt / Confidence F1 Sweep",
        )
        save_prompt_sweep_plot(
            prompt_f2_plot,
            prompt_sweep_rows,
            metric="f2",
            title="SAM 3 MinneApple Validation — Prompt / Confidence F2 Sweep",
        )

        # ------------------------------------------------------------------
        # 6/7 Stage B — lightweight post-processing grid
        # ------------------------------------------------------------------
        print()
        print("[6/7] Stage B: mask-area + duplicate-suppression optimization...")

        config_product = list(
            itertools.product(
                top_prompts,
                thresholds,
                min_area_pixels,
                max_area_fractions,
                nms_ious,
            )
        )
        print(f"  Configurations       : {len(config_product)}")

        postprocess_rows: list[dict] = []

        for index, (
            prompt,
            threshold,
            min_area,
            max_fraction,
            nms_iou,
        ) in enumerate(config_product, start=1):
            metrics, _ = evaluate_config(
                records_by_prompt[prompt],
                confidence_threshold=threshold,
                min_area_pixels=min_area,
                max_area_fraction=max_fraction,
                nms_iou=nms_iou,
                match_iou=args.match_iou,
                accept_ious=accept_ious,
                include_per_image=False,
            )
            postprocess_rows.append(metrics)

            if index % 100 == 0 or index == len(config_product):
                current_best = best_row(postprocess_rows, args.objective)
                print(
                    f"  Search progress       : {index}/{len(config_product)} | "
                    f"best prompt={current_best['prompt']!r} "
                    f"conf={current_best['confidence_threshold']:.2f} "
                    f"F1={current_best['f1']:.4f} F2={current_best['f2']:.4f} "
                    f"actions={current_best['review_actions_proxy']}"
                )

        selected = best_row(postprocess_rows, args.objective)
        best_f1 = best_row(postprocess_rows, "f1")
        best_f2 = best_row(postprocess_rows, "f2")
        best_actions = best_row(postprocess_rows, "review_actions")

        sorted_postprocess = sorted(
            postprocess_rows,
            key=lambda row: config_rank_key(row, args.objective),
            reverse=True,
        )

        postprocess_csv_rows = []
        for rank, row in enumerate(sorted_postprocess, start=1):
            flattened = flatten_metric_row(row, accept_ious)
            flattened = {"rank": rank, **flattened}
            postprocess_csv_rows.append(flattened)

        write_csv(
            postprocess_sweep_csv,
            postprocess_csv_rows,
            ["rank"] + metric_fieldnames(accept_ious),
        )

        # Per-image metrics for selected configuration.
        selected_metrics, selected_per_image = evaluate_config(
            records_by_prompt[selected["prompt"]],
            confidence_threshold=selected["confidence_threshold"],
            min_area_pixels=selected["min_area_pixels"],
            max_area_fraction=selected["max_area_fraction"],
            nms_iou=selected["mask_nms_iou"],
            match_iou=args.match_iou,
            accept_ious=accept_ious,
            include_per_image=True,
        )

        per_image_fields = [
            "sample_id",
            "prompt",
            "confidence_threshold",
            "min_area_pixels",
            "max_area_fraction",
            "mask_nms_iou",
            "gt_instances",
            "cached_predictions",
            "kept_predictions",
            "tp",
            "fp_delete",
            "fn_add",
            "refine_actions",
            "review_actions_proxy",
            "precision",
            "recall",
            "f1",
            "f2",
            "mean_matched_iou",
        ]
        for threshold in accept_ious:
            per_image_fields.append(f"auto_accept_{str(threshold).replace('.', '_')}")

        write_csv(selected_per_image_csv, selected_per_image, per_image_fields)

        filtered_ap50 = compute_filtered_average_precision(
            records_by_prompt[selected["prompt"]],
            min_area_pixels=selected["min_area_pixels"],
            max_area_fraction=selected["max_area_fraction"],
            nms_iou=selected["mask_nms_iou"],
            base_confidence=args.base_confidence,
            iou_threshold=0.50,
        )
        filtered_ap75 = compute_filtered_average_precision(
            records_by_prompt[selected["prompt"]],
            min_area_pixels=selected["min_area_pixels"],
            max_area_fraction=selected["max_area_fraction"],
            nms_iou=selected["mask_nms_iou"],
            base_confidence=args.base_confidence,
            iou_threshold=0.75,
        )

        # ------------------------------------------------------------------
        # 7/7 Save final report and baseline comparison
        # ------------------------------------------------------------------
        print()
        print("[7/7] Saving optimization report...")

        baseline_summary = None
        baseline_comparison = None

        if baseline_summary_path.is_file():
            try:
                baseline_summary = read_json(baseline_summary_path)
                baseline_best = baseline_summary.get("best_validation_metrics", {})
                if baseline_best:
                    baseline_comparison = {
                        "baseline_summary": relative_or_absolute(
                            baseline_summary_path, project_root
                        ),
                        "baseline_prompt": baseline_summary.get("prompt"),
                        "baseline_best_threshold": baseline_summary.get(
                            "best_validation_threshold_by_f1"
                        ),
                        "baseline_precision": baseline_best.get("precision"),
                        "baseline_recall": baseline_best.get("recall"),
                        "baseline_f1": baseline_best.get("f1"),
                        "baseline_mean_matched_iou": baseline_best.get(
                            "mean_matched_iou"
                        ),
                        "selected_precision": selected_metrics["precision"],
                        "selected_recall": selected_metrics["recall"],
                        "selected_f1": selected_metrics["f1"],
                        "selected_mean_matched_iou": selected_metrics[
                            "mean_matched_iou"
                        ],
                        "delta_precision": selected_metrics["precision"]
                        - float(baseline_best.get("precision", 0.0)),
                        "delta_recall": selected_metrics["recall"]
                        - float(baseline_best.get("recall", 0.0)),
                        "delta_f1": selected_metrics["f1"]
                        - float(baseline_best.get("f1", 0.0)),
                        "delta_mean_matched_iou": selected_metrics[
                            "mean_matched_iou"
                        ]
                        - float(baseline_best.get("mean_matched_iou", 0.0)),
                    }
            except Exception as error:
                summary["warnings"].append(
                    f"Could not load baseline summary for comparison: {error}"
                )

        summary["stage_a_prompt_sweep"] = {
            "best_per_prompt": prompt_best_rows,
            "top_prompts_for_postprocess": top_prompts,
        }
        summary["stage_b_postprocess_search"] = {
            "configurations_evaluated": len(postprocess_rows),
            "selected_by_objective": selected_metrics,
            "best_by_f1": best_f1,
            "best_by_f2": best_f2,
            "best_by_review_actions": best_actions,
        }
        summary["selected_configuration"] = {
            "objective": args.objective,
            "prompt": selected["prompt"],
            "confidence_threshold": selected["confidence_threshold"],
            "min_area_pixels": selected["min_area_pixels"],
            "max_area_fraction": selected["max_area_fraction"],
            "mask_nms_iou": selected["mask_nms_iou"],
        }
        summary["selected_metrics"] = selected_metrics
        summary["filtered_mask_ap"] = {
            "note": (
                "AP uses the selected structural filters (mask area + mask NMS) "
                "but ranks all candidates available at base_confidence by score."
            ),
            "base_confidence": args.base_confidence,
            "ap50": filtered_ap50,
            "ap75": filtered_ap75,
        }
        summary["baseline_comparison"] = baseline_comparison
        summary["outputs"] = {
            "experiment_root": relative_or_absolute(optimization_root, project_root),
            "cache_index_csv": relative_or_absolute(cache_index_csv, project_root),
            "prompt_sweep_csv": relative_or_absolute(prompt_sweep_csv, project_root),
            "postprocess_sweep_csv": relative_or_absolute(
                postprocess_sweep_csv, project_root
            ),
            "selected_per_image_csv": relative_or_absolute(
                selected_per_image_csv, project_root
            ),
            "prompt_f1_plot": relative_or_absolute(prompt_f1_plot, project_root)
            if prompt_f1_plot.exists()
            else None,
            "prompt_f2_plot": relative_or_absolute(prompt_f2_plot, project_root)
            if prompt_f2_plot.exists()
            else None,
            "summary_json": relative_or_absolute(summary_path, project_root),
        }

        summary["status"] = "PASS_WITH_WARNINGS" if summary["warnings"] else "PASS"
        write_json(summary_path, summary)

        strict_accept = max(accept_ious)
        strict_label = str(strict_accept).replace(".", "_")

        print()
        print("=" * 96)
        print("SAM 3 ZERO-SHOT OPTIMIZATION COMPLETE")
        print("=" * 96)
        print(f"STATUS                         : {summary['status']}")
        print(f"SPLIT                          : {args.split}")
        print(f"IMAGES                         : {len(selected_rows)}")
        print(f"GT APPLE INSTANCES             : {total_manifest_gt}")
        print(f"OBJECTIVE                      : {args.objective}")
        print(f"SELECTED PROMPT                : {selected_metrics['prompt']}")
        print(
            f"SELECTED CONFIDENCE            : "
            f"{selected_metrics['confidence_threshold']:.2f}"
        )
        print(f"MIN MASK AREA                  : {selected_metrics['min_area_pixels']} px")
        print(
            f"MAX MASK AREA FRACTION         : "
            f"{selected_metrics['max_area_fraction']:.4f}"
        )
        print(
            f"MASK NMS IoU                   : "
            f"{selected_metrics['mask_nms_iou']:.2f} (0=off)"
        )
        print(
            f"PRECISION @IoU{args.match_iou:.2f}              : "
            f"{selected_metrics['precision']:.4f}"
        )
        print(
            f"RECALL @IoU{args.match_iou:.2f}                 : "
            f"{selected_metrics['recall']:.4f}"
        )
        print(f"F1                              : {selected_metrics['f1']:.4f}")
        print(f"F2                              : {selected_metrics['f2']:.4f}")
        print(
            f"MEAN MATCHED MASK IoU           : "
            f"{selected_metrics['mean_matched_iou']:.4f}"
        )
        print(
            f"AUTO ACCEPT @IoU{strict_accept:.2f}           : "
            f"{selected_metrics[f'auto_accept_rate_gt_{strict_label}']:.4f}"
        )
        print(
            f"REVIEW ACTIONS PROXY            : "
            f"{selected_metrics['review_actions_proxy']} "
            f"({selected_metrics['review_actions_per_image']:.2f}/image)"
        )
        print(f"FILTERED MASK AP50              : {filtered_ap50['ap']:.4f}")
        print(f"FILTERED MASK AP75              : {filtered_ap75['ap']:.4f}")

        if baseline_comparison:
            print(
                f"DELTA F1 VS RAW BASELINE        : "
                f"{baseline_comparison['delta_f1']:+.4f}"
            )
            print(
                f"DELTA RECALL VS RAW BASELINE    : "
                f"{baseline_comparison['delta_recall']:+.4f}"
            )
            print(
                f"DELTA PRECISION VS RAW BASELINE : "
                f"{baseline_comparison['delta_precision']:+.4f}"
            )

        print(f"PROMPT SWEEP                    : {prompt_sweep_csv}")
        print(f"POSTPROCESS SWEEP               : {postprocess_sweep_csv}")
        print(f"PER-IMAGE METRICS               : {selected_per_image_csv}")
        print(f"SUMMARY                         : {summary_path}")
        print("=" * 96)

    except Exception as error:
        summary["status"] = "FAIL"
        summary["errors"].append(str(error))
        summary["exception_type"] = type(error).__name__
        summary["traceback"] = traceback.format_exc()
        write_json(summary_path, summary)

        print()
        print("=" * 96)
        print("SAM 3 ZERO-SHOT OPTIMIZATION FAILED")
        print("=" * 96)
        print(f"ERROR   : {error}")
        print(f"REPORT  : {summary_path}")
        print("=" * 96)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
