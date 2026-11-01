from __future__ import annotations

from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from importlib import metadata as importlib_metadata
from collections import defaultdict

import argparse
import csv
import hashlib
import itertools
import json
import platform
import random
import re
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw


# =============================================================================
# SAM 3 Occlusion / Multi-Scale Experiment for AgriAnnotationSystem
# =============================================================================
#
# Goal
# ----
# Improve apple discovery in difficult orchard scenes, especially for:
#   - partially visible apples
#   - leaf-occluded apples
#   - small / distant apples
#   - dense apple clusters
#
# The script compares three inference strategies on the frozen validation split:
#
#   A. FULL_OPTIMIZED
#      Current optimized zero-shot configuration from Stage 4B.
#
#   B. FULL_PLUS_TILES
#      Full-image SAM 3 + overlapping tiled SAM 3 using "orchard apple".
#
#   C. FULL_PLUS_TILES_MULTI_PROMPT
#      Full-image SAM 3 + overlapping tile inference using several
#      occlusion-aware prompts.
#
# IMPORTANT TASK DEFINITION
# -------------------------
# This project evaluates VISIBLE instance segmentation. The model should segment
# only visible apple pixels. It does NOT hallucinate the hidden part of an apple.
# Predicting hidden geometry would be an amodal-segmentation task and is outside
# the scope of this experiment.
#
# Main metrics
# ------------
#   - Precision / Recall / F1 / F2 @ mask IoU >= 0.50
#   - Mean matched mask IoU
#   - AutoAccept rates @ IoU 0.70 / 0.75 / 0.80 / 0.85
#   - Human-review action proxy: Delete + Add + Refine
#   - Recall stratified by visible GT mask area quartile
#   - Recovered / lost GT instances relative to FULL_OPTIMIZED
#
# Why F2?
# -------
# Occlusion recovery is recall-sensitive. In an annotation system, deleting an
# extra false positive is usually cheaper than manually discovering and adding a
# missed apple. Therefore the default selection objective is F2, while F1 and
# review-action optima are also reported.
#
# Cache / resume
# --------------
# All expensive SAM 3 inference is cached per image and per tile-prompt. The
# script is safe to rerun. Existing compatible caches are reused unless
# --overwrite is supplied.
#
# Official test protection
# ------------------------
# The MinneApple official test split remains frozen. This script refuses to run
# on split=test unless --allow-test is explicitly supplied.
#
# =============================================================================


MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)
MIN_CUDA = (12, 6)

MANIFEST_RELATIVE_PATH = Path("data/splits/minneapple_split.csv")
DEFAULT_CHECKPOINT_RELATIVE_PATH = Path("models/sam3/checkpoints/sam3.pt")
THIRD_PARTY_SAM3_RELATIVE_PATH = Path("third_party/sam3")
OUTPUTS_RELATIVE_PATH = Path("outputs/experiments")

# Stage-4B defaults frozen from the user's validation optimization.
DEFAULT_FULL_PROMPT = "orchard apple"
DEFAULT_FULL_CONFIDENCE = 0.45
DEFAULT_BASE_CONFIDENCE = 0.25
DEFAULT_MIN_AREA_PIXELS = 200
DEFAULT_MAX_AREA_FRACTION = 0.01
DEFAULT_MASK_NMS_IOU = 0.70

# Multi-scale defaults.
DEFAULT_GRID_ROWS = 2
DEFAULT_GRID_COLS = 2
DEFAULT_TILE_OVERLAP = 0.20
DEFAULT_TILE_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45]
DEFAULT_TILE_MIN_AREAS = [25, 50, 100, 200]
DEFAULT_FUSION_NMS_IOUS = [0.60, 0.70, 0.80, 0.90]

# Prompt set for the tile-based occlusion-aware strategy.
DEFAULT_OCCLUSION_PROMPTS = [
    "orchard apple",
    "partially visible apple",
    "apple behind leaves",
    "occluded apple",
]

DEFAULT_MATCH_IOU = 0.50
DEFAULT_ACCEPT_IOUS = [0.70, 0.75, 0.80, 0.85]
DEFAULT_OBJECTIVE = "f2"


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


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def fbeta_score(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denominator)


def resolve_project_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    # Expected location:
    # AgriAnnotationSystem/scripts/04c_occlusion_multiscale.py
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
    raw = values if values else default
    parsed = sorted(set(round(float(value), 8) for value in raw))

    for value in parsed:
        if value < minimum or value > maximum:
            raise SystemExit(f"{name}: value {value} is outside [{minimum}, {maximum}].")
        if not allow_zero and value == 0:
            raise SystemExit(f"{name}: zero is not allowed.")

    return parsed


def parse_unique_ints(
    values: list[int] | None,
    default: list[int],
    name: str,
    minimum: int,
) -> list[int]:
    raw = values if values else default
    parsed = sorted(set(int(value) for value in raw))

    for value in parsed:
        if value < minimum:
            raise SystemExit(f"{name}: value {value} must be >= {minimum}.")

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

        if torch.version.cuda is not None:
            if version_tuple(torch.version.cuda, 2) < MIN_CUDA:
                errors.append(
                    f"CUDA {MIN_CUDA[0]}.{MIN_CUDA[1]}+ is recommended; "
                    f"PyTorch reports CUDA {torch.version.cuda}."
                )

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            environment.update(
                {
                    "cuda_device_index": int(index),
                    "cuda_device_name": torch.cuda.get_device_name(index),
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
# Manifest / ground truth
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


def load_subset_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Subset CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows or "sample_id" not in rows[0]:
        raise RuntimeError("Subset CSV must contain a 'sample_id' column.")

    return {row["sample_id"].strip() for row in rows if row.get("sample_id", "").strip()}


def select_split_rows(
    rows: list[dict[str, str]],
    split: str,
    limit: int | None,
    subset_ids: set[str] | None,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("split", "").strip().lower() == split.strip().lower()
    ]

    if subset_ids is not None:
        selected = [row for row in selected if row.get("sample_id") in subset_ids]

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
# Tensor / SAM conversion
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
            "SAM 3 mask size does not match image: "
            f"masks={masks.shape[1:]}, image={(expected_height, expected_width)}"
        )

    return masks.astype(bool, copy=False)


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


# =============================================================================
# RLE helpers
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
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": counts,
    }


def decode_rle(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_utils

    decoded = mask_utils.decode(coco_rle(rle))
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return decoded.astype(bool, copy=False)


def gt_rles_from_indexed(mask: np.ndarray, ids: np.ndarray) -> list[dict]:
    return [encode_binary_mask(mask == instance_id) for instance_id in ids]


def rle_area(rle: dict) -> float:
    from pycocotools import mask as mask_utils

    value = mask_utils.area(coco_rle(rle))
    return float(np.asarray(value).reshape(-1)[0])


def rle_iou_matrix(dt_rles: list[dict], gt_rles: list[dict]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    if not dt_rles or not gt_rles:
        return np.zeros((len(dt_rles), len(gt_rles)), dtype=np.float32)

    dt = [coco_rle(rle) for rle in dt_rles]
    gt = [coco_rle(rle) for rle in gt_rles]
    matrix = mask_utils.iou(dt, gt, [0] * len(gt))
    return np.asarray(matrix, dtype=np.float32)


def pred_pairwise_iou_matrix(pred_rles: list[dict]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    if not pred_rles:
        return np.zeros((0, 0), dtype=np.float16)

    rles = [coco_rle(rle) for rle in pred_rles]
    matrix = mask_utils.iou(rles, rles, [0] * len(rles))
    return np.asarray(matrix, dtype=np.float16)


# =============================================================================
# Tile generation and coordinate mapping
# =============================================================================


def generate_overlapping_tiles(
    width: int,
    height: int,
    rows: int,
    cols: int,
    overlap_fraction: float,
) -> list[dict]:
    """
    Generate a regular grid where each nominal cell is expanded toward its
    neighbors. overlap_fraction=0.20 means adjacent tiles overlap by roughly
    20% of a nominal cell dimension.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("Tile rows/cols must be > 0.")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("Tile overlap must be in [0, 1).")

    cell_w = width / cols
    cell_h = height / rows
    expand_x = cell_w * overlap_fraction / 2.0
    expand_y = cell_h * overlap_fraction / 2.0

    tiles: list[dict] = []
    tile_index = 0

    for row in range(rows):
        for col in range(cols):
            nominal_x0 = col * cell_w
            nominal_x1 = (col + 1) * cell_w
            nominal_y0 = row * cell_h
            nominal_y1 = (row + 1) * cell_h

            x0 = int(round(max(0.0, nominal_x0 - (expand_x if col > 0 else 0.0))))
            x1 = int(round(min(float(width), nominal_x1 + (expand_x if col < cols - 1 else 0.0))))
            y0 = int(round(max(0.0, nominal_y0 - (expand_y if row > 0 else 0.0))))
            y1 = int(round(min(float(height), nominal_y1 + (expand_y if row < rows - 1 else 0.0))))

            if x1 <= x0 or y1 <= y0:
                continue

            tiles.append(
                {
                    "tile_index": tile_index,
                    "row": row,
                    "col": col,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "width": x1 - x0,
                    "height": y1 - y0,
                }
            )
            tile_index += 1

    return tiles


def map_tile_predictions_to_full(
    tile_masks: np.ndarray,
    tile_boxes: np.ndarray,
    scores: np.ndarray,
    tile: dict,
    full_width: int,
    full_height: int,
    prompt: str,
) -> list[dict]:
    records: list[dict] = []

    x0 = int(tile["x0"])
    y0 = int(tile["y0"])
    x1 = int(tile["x1"])
    y1 = int(tile["y1"])

    for index in range(len(tile_masks)):
        tile_mask = tile_masks[index].astype(bool, copy=False)
        full_mask = np.zeros((full_height, full_width), dtype=bool)
        full_mask[y0:y1, x0:x1] = tile_mask

        box = tile_boxes[index]
        full_box = [
            float(np.clip(box[0] + x0, 0, full_width)),
            float(np.clip(box[1] + y0, 0, full_height)),
            float(np.clip(box[2] + x0, 0, full_width)),
            float(np.clip(box[3] + y0, 0, full_height)),
        ]

        # A mask touching an internal tile boundary may represent a truncated
        # object. We keep it (important for recall) but record the flag so the
        # result can be inspected later.
        touches_left = bool(tile_mask[:, 0].any()) and x0 > 0
        touches_right = bool(tile_mask[:, -1].any()) and x1 < full_width
        touches_top = bool(tile_mask[0, :].any()) and y0 > 0
        touches_bottom = bool(tile_mask[-1, :].any()) and y1 < full_height
        touches_internal_edge = touches_left or touches_right or touches_top or touches_bottom

        records.append(
            {
                "class_label": "apple",
                "prompt": prompt,
                "source": "tile",
                "tile_index": int(tile["tile_index"]),
                "tile_row": int(tile["row"]),
                "tile_col": int(tile["col"]),
                "tile_xyxy": [x0, y0, x1, y1],
                "touches_internal_tile_edge": bool(touches_internal_edge),
                "score": float(scores[index]),
                "box_xyxy": full_box,
                "mask_area_pixels": int(full_mask.sum()),
                "rle": encode_binary_mask(full_mask),
            }
        )

    return records


# =============================================================================
# Prediction cache
# =============================================================================


def full_cache_signature(sample: dict[str, str], prompt: str, base_confidence: float) -> dict:
    return {
        "sample_id": sample.get("sample_id"),
        "image_path": sample.get("processed_image_path"),
        "mask_path": sample.get("processed_mask_path"),
        "prompt": prompt,
        "base_confidence": round(float(base_confidence), 6),
    }


def tile_cache_signature(
    sample: dict[str, str],
    prompt: str,
    base_confidence: float,
    rows: int,
    cols: int,
    overlap: float,
) -> dict:
    return {
        "sample_id": sample.get("sample_id"),
        "image_path": sample.get("processed_image_path"),
        "prompt": prompt,
        "base_confidence": round(float(base_confidence), 6),
        "grid_rows": int(rows),
        "grid_cols": int(cols),
        "tile_overlap": round(float(overlap), 6),
    }


def cache_is_compatible(data: dict, signature: dict) -> bool:
    return data.get("status") == "PASS" and data.get("signature") == signature


def full_local_cache_path(experiment_root: Path, prompt: str, sample_id: str) -> Path:
    return experiment_root / "predictions" / "full" / prompt_slug(prompt) / f"{sample_id}.json"


def tile_cache_path(experiment_root: Path, prompt: str, sample_id: str) -> Path:
    return experiment_root / "predictions" / "tiles" / prompt_slug(prompt) / f"{sample_id}.json"


def stage4b_full_cache_path(
    project_root: Path,
    split: str,
    prompt: str,
    sample_id: str,
) -> Path:
    return (
        project_root
        / OUTPUTS_RELATIVE_PATH
        / f"sam3_zero_shot_optimization_{split}"
        / "predictions"
        / prompt_slug(prompt)
        / f"{sample_id}.json"
    ).resolve()


def load_compatible_full_cache(
    project_root: Path,
    experiment_root: Path,
    split: str,
    sample: dict[str, str],
    prompt: str,
    base_confidence: float,
) -> tuple[Path | None, str]:
    signature = full_cache_signature(sample, prompt, base_confidence)

    candidates = [
        (full_local_cache_path(experiment_root, prompt, sample["sample_id"]), "local"),
        (stage4b_full_cache_path(project_root, split, prompt, sample["sample_id"]), "stage4b"),
    ]

    for path, source in candidates:
        if not path.is_file():
            continue
        try:
            data = read_json(path)
            if cache_is_compatible(data, signature):
                return path, source
        except Exception:
            continue

    return None, "missing"


def load_compatible_tile_cache(
    experiment_root: Path,
    sample: dict[str, str],
    prompt: str,
    base_confidence: float,
    rows: int,
    cols: int,
    overlap: float,
) -> Path | None:
    path = tile_cache_path(experiment_root, prompt, sample["sample_id"])
    if not path.is_file():
        return None

    signature = tile_cache_signature(sample, prompt, base_confidence, rows, cols, overlap)
    try:
        data = read_json(path)
        if cache_is_compatible(data, signature):
            return path
    except Exception:
        pass

    return None


def save_full_cache(
    path: Path,
    signature: dict,
    sample: dict[str, str],
    image_size: tuple[int, int],
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
            "sample_id": sample["sample_id"],
            "split": sample.get("split"),
            "image_path": sample.get("processed_image_path"),
            "mask_path": sample.get("processed_mask_path"),
            "width": int(image_size[0]),
            "height": int(image_size[1]),
        },
        "inference": {
            "seconds": round(float(inference_seconds), 6),
            "autocast": autocast_label,
            "predicted_instances": len(predictions),
        },
        "predictions": predictions,
    }
    write_json(path, payload)


def save_tile_cache(
    path: Path,
    signature: dict,
    sample: dict[str, str],
    image_size: tuple[int, int],
    tiles: list[dict],
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
            "sample_id": sample["sample_id"],
            "split": sample.get("split"),
            "image_path": sample.get("processed_image_path"),
            "width": int(image_size[0]),
            "height": int(image_size[1]),
        },
        "tiles": tiles,
        "inference": {
            "seconds": round(float(inference_seconds), 6),
            "autocast": autocast_label,
            "predicted_instances": len(predictions),
        },
        "predictions": predictions,
    }
    write_json(path, payload)


def normalize_cached_predictions(
    predictions: list[dict],
    source: str,
    prompt_fallback: str,
) -> list[dict]:
    normalized: list[dict] = []
    for item in predictions:
        record = dict(item)
        record["source"] = record.get("source", source)
        record["prompt"] = record.get("prompt", prompt_fallback)
        record["class_label"] = record.get("class_label", "apple")
        normalized.append(record)
    return normalized


# =============================================================================
# Build evaluation record
# =============================================================================


def load_strategy_record(
    project_root: Path,
    sample: dict[str, str],
    full_cache: Path,
    tile_caches: list[tuple[str, Path]],
) -> dict:
    mask_path = resolve_row_path(project_root, sample["processed_mask_path"])
    gt_mask, gt_ids = load_gt_indexed_mask(mask_path)
    gt_rles = gt_rles_from_indexed(gt_mask, gt_ids)
    gt_areas = np.asarray([rle_area(rle) for rle in gt_rles], dtype=np.float32)

    full_data = read_json(full_cache)
    full_prompt = full_data.get("signature", {}).get("prompt", DEFAULT_FULL_PROMPT)
    predictions = normalize_cached_predictions(
        full_data.get("predictions", []),
        source="full",
        prompt_fallback=full_prompt,
    )

    inference_seconds = float(full_data.get("inference", {}).get("seconds", 0.0))

    for tile_prompt, path in tile_caches:
        data = read_json(path)
        predictions.extend(
            normalize_cached_predictions(
                data.get("predictions", []),
                source="tile",
                prompt_fallback=tile_prompt,
            )
        )
        inference_seconds += float(data.get("inference", {}).get("seconds", 0.0))

    scores = np.asarray([float(item["score"]) for item in predictions], dtype=np.float32)
    areas = np.asarray(
        [int(item.get("mask_area_pixels", 0)) for item in predictions],
        dtype=np.int64,
    )
    sources = np.asarray([item.get("source", "full") for item in predictions], dtype=object)
    prompts = np.asarray([item.get("prompt", "") for item in predictions], dtype=object)
    pred_rles = [item["rle"] for item in predictions]

    gt_iou = rle_iou_matrix(pred_rles, gt_rles)
    pred_iou = pred_pairwise_iou_matrix(pred_rles)

    height, width = gt_mask.shape

    return {
        "sample_id": sample["sample_id"],
        "image_path": sample["processed_image_path"],
        "mask_path": sample["processed_mask_path"],
        "gt_count": int(len(gt_ids)),
        "gt_ids": gt_ids.astype(np.int64),
        "gt_areas": gt_areas,
        "gt_rles": gt_rles,
        "predictions": predictions,
        "pred_count": int(len(predictions)),
        "scores": scores,
        "areas": areas,
        "sources": sources,
        "prompts": prompts,
        "image_area": int(height * width),
        "image_width": int(width),
        "image_height": int(height),
        "gt_iou_matrix": gt_iou,
        "pred_iou_matrix": pred_iou,
        "inference_seconds": inference_seconds,
    }


# =============================================================================
# Selection, NMS, matching
# =============================================================================


def select_prediction_indices(
    record: dict,
    full_confidence: float,
    tile_confidence: float,
    min_area_pixels: int,
    max_area_fraction: float,
    nms_iou: float,
) -> np.ndarray:
    scores = record["scores"]
    areas = record["areas"]
    sources = record["sources"]
    max_area_pixels = float(record["image_area"]) * float(max_area_fraction)

    thresholds = np.where(sources == "full", full_confidence, tile_confidence)

    keep = np.where(
        (scores >= thresholds)
        & (areas >= int(min_area_pixels))
        & (areas <= max_area_pixels)
    )[0]

    if len(keep) == 0:
        return np.empty((0,), dtype=np.int64)

    # Highest score first. A tiny preference for full-image predictions is added
    # only as a deterministic tie breaker; it does not alter normal score order.
    source_bonus = np.where(sources[keep] == "full", 1e-6, 0.0)
    order = keep[np.argsort(scores[keep] + source_bonus)[::-1]]

    if nms_iou <= 0.0:
        return order.astype(np.int64)

    pairwise = record["pred_iou_matrix"]
    selected: list[int] = []

    # Vectorized overlap check against already-selected masks. This matters for
    # the multi-prompt strategy, where one image can contain hundreds of cached
    # candidates. Avoiding the Python inner loop keeps the validation sweep fast.
    for pred_index in order:
        pred_index = int(pred_index)
        if selected:
            overlaps = np.asarray(pairwise[pred_index, selected], dtype=np.float32)
            if bool(np.any(overlaps >= nms_iou)):
                continue
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
        "matched_gt_indices": sorted(matched_gt),
    }


def evaluate_records(
    records: list[dict],
    strategy_name: str,
    full_confidence: float,
    tile_confidence: float,
    min_area_pixels: int,
    max_area_fraction: float,
    nms_iou: float,
    match_iou: float,
    accept_ious: list[float],
    include_per_image: bool = False,
) -> tuple[dict, list[dict], dict[str, set[int]]]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_selected = 0
    all_matched_ious: list[float] = []
    accept_counts = {threshold: 0 for threshold in accept_ious}
    per_image_rows: list[dict] = []
    matched_gt_by_image: dict[str, set[int]] = {}

    strict_accept = max(accept_ious)

    for record in records:
        selected = select_prediction_indices(
            record,
            full_confidence=full_confidence,
            tile_confidence=tile_confidence,
            min_area_pixels=min_area_pixels,
            max_area_fraction=max_area_fraction,
            nms_iou=nms_iou,
        )

        matched = match_selected_predictions(
            record["gt_iou_matrix"],
            selected,
            record["scores"],
            match_iou,
        )

        matched_set = set(int(value) for value in matched["matched_gt_indices"])
        matched_gt_by_image[record["sample_id"]] = matched_set

        total_tp += matched["tp"]
        total_fp += matched["fp"]
        total_fn += matched["fn"]
        total_gt += record["gt_count"]
        total_selected += matched["selected_predictions"]
        all_matched_ious.extend(matched["matched_ious"])

        local_accept: dict[float, int] = {}
        for threshold in accept_ious:
            count = sum(1 for value in matched["matched_ious"] if value >= threshold)
            accept_counts[threshold] += count
            local_accept[threshold] = count

        precision = safe_divide(matched["tp"], matched["tp"] + matched["fp"])
        recall = safe_divide(matched["tp"], matched["tp"] + matched["fn"])
        f1 = fbeta_score(precision, recall, 1.0)
        f2 = fbeta_score(precision, recall, 2.0)

        strict_auto = local_accept[strict_accept]
        refine_actions = max(0, matched["tp"] - strict_auto)
        review_actions = matched["fp"] + matched["fn"] + refine_actions

        if include_per_image:
            per_image_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "strategy": strategy_name,
                    "gt_instances": record["gt_count"],
                    "candidate_predictions": record["pred_count"],
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
                    "mean_matched_iou": round(float(np.mean(matched["matched_ious"])), 6)
                    if matched["matched_ious"]
                    else 0.0,
                    "inference_seconds": round(float(record["inference_seconds"]), 6),
                }
            )

    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)
    f1 = fbeta_score(precision, recall, 1.0)
    f2 = fbeta_score(precision, recall, 2.0)
    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    strict_auto = accept_counts[strict_accept]
    refine_actions = max(0, total_tp - strict_auto)
    review_actions = total_fp + total_fn + refine_actions

    metrics = {
        "strategy": strategy_name,
        "full_confidence": float(full_confidence),
        "tile_confidence": float(tile_confidence),
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
        "mean_inference_seconds_per_image": float(
            np.mean([record["inference_seconds"] for record in records])
        )
        if records
        else 0.0,
    }

    for threshold in accept_ious:
        label = str(threshold).replace(".", "_")
        count = accept_counts[threshold]
        metrics[f"auto_accept_tp_{label}"] = int(count)
        metrics[f"auto_accept_rate_gt_{label}"] = safe_divide(count, total_gt)

    return metrics, per_image_rows, matched_gt_by_image


# =============================================================================
# Strategy search / ranking
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
        raise RuntimeError("No metric rows were generated.")
    return max(rows, key=lambda row: config_rank_key(row, objective))


def search_enhanced_strategy(
    records: list[dict],
    strategy_name: str,
    full_confidence: float,
    tile_thresholds: list[float],
    min_areas: list[int],
    max_area_fraction: float,
    nms_ious: list[float],
    match_iou: float,
    accept_ious: list[float],
) -> list[dict]:
    rows: list[dict] = []

    combinations = list(itertools.product(tile_thresholds, min_areas, nms_ious))
    total = len(combinations)

    for index, (tile_confidence, min_area, nms_iou) in enumerate(combinations, start=1):
        metrics, _, _ = evaluate_records(
            records=records,
            strategy_name=strategy_name,
            full_confidence=full_confidence,
            tile_confidence=tile_confidence,
            min_area_pixels=min_area,
            max_area_fraction=max_area_fraction,
            nms_iou=nms_iou,
            match_iou=match_iou,
            accept_ious=accept_ious,
            include_per_image=False,
        )
        rows.append(metrics)

        if index % 20 == 0 or index == total:
            print(
                f"    {strategy_name}: searched {index}/{total} | "
                f"tile_conf={tile_confidence:.2f} min_area={min_area} nms={nms_iou:.2f} "
                f"F1={metrics['f1']:.4f} F2={metrics['f2']:.4f} R={metrics['recall']:.4f}"
            )

    return rows


# =============================================================================
# GT area-stratified recall and recovery analysis
# =============================================================================


def compute_gt_area_quartile_edges(records: list[dict]) -> tuple[float, float, float]:
    all_areas: list[float] = []
    for record in records:
        all_areas.extend(float(value) for value in record["gt_areas"])

    if not all_areas:
        return 0.0, 0.0, 0.0

    q1, q2, q3 = np.quantile(np.asarray(all_areas, dtype=np.float32), [0.25, 0.50, 0.75])
    return float(q1), float(q2), float(q3)


def area_bin_label(area: float, edges: tuple[float, float, float]) -> str:
    q1, q2, q3 = edges
    if area <= q1:
        return "Q1_smallest_visible"
    if area <= q2:
        return "Q2_small"
    if area <= q3:
        return "Q3_medium"
    return "Q4_largest_visible"


def stratified_recall_rows(
    records: list[dict],
    matched_gt_by_image: dict[str, set[int]],
    strategy_name: str,
    edges: tuple[float, float, float],
) -> list[dict]:
    counts = defaultdict(lambda: {"gt": 0, "matched": 0})

    for record in records:
        matched = matched_gt_by_image.get(record["sample_id"], set())
        for gt_index, area in enumerate(record["gt_areas"]):
            label = area_bin_label(float(area), edges)
            counts[label]["gt"] += 1
            if gt_index in matched:
                counts[label]["matched"] += 1

    order = [
        "Q1_smallest_visible",
        "Q2_small",
        "Q3_medium",
        "Q4_largest_visible",
    ]

    rows = []
    for label in order:
        gt = counts[label]["gt"]
        matched = counts[label]["matched"]
        rows.append(
            {
                "strategy": strategy_name,
                "visible_area_bin": label,
                "gt_instances": gt,
                "matched_instances": matched,
                "recall": safe_divide(matched, gt),
            }
        )
    return rows


def recovery_rows(
    records: list[dict],
    baseline_matches: dict[str, set[int]],
    enhanced_matches: dict[str, set[int]],
    enhanced_strategy: str,
    edges: tuple[float, float, float],
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    recovered_by_bin = defaultdict(int)
    lost_by_bin = defaultdict(int)
    total_recovered = 0
    total_lost = 0

    for record in records:
        sample_id = record["sample_id"]
        baseline = baseline_matches.get(sample_id, set())
        enhanced = enhanced_matches.get(sample_id, set())

        recovered = sorted(enhanced - baseline)
        lost = sorted(baseline - enhanced)

        for gt_index in recovered:
            area = float(record["gt_areas"][gt_index])
            label = area_bin_label(area, edges)
            recovered_by_bin[label] += 1
            total_recovered += 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "change": "recovered",
                    "gt_index": gt_index,
                    "gt_visible_area_pixels": area,
                    "visible_area_bin": label,
                    "enhanced_strategy": enhanced_strategy,
                }
            )

        for gt_index in lost:
            area = float(record["gt_areas"][gt_index])
            label = area_bin_label(area, edges)
            lost_by_bin[label] += 1
            total_lost += 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "change": "lost",
                    "gt_index": gt_index,
                    "gt_visible_area_pixels": area,
                    "visible_area_bin": label,
                    "enhanced_strategy": enhanced_strategy,
                }
            )

    summary = {
        "enhanced_strategy": enhanced_strategy,
        "recovered_gt_instances": total_recovered,
        "lost_gt_instances": total_lost,
        "net_gt_recovery": total_recovered - total_lost,
        "recovered_by_visible_area_bin": dict(recovered_by_bin),
        "lost_by_visible_area_bin": dict(lost_by_bin),
        "note": (
            "Visible-area quartiles are a proxy for difficult/small visible objects. "
            "They are not a direct occlusion annotation."
        ),
    }

    return rows, summary


# =============================================================================
# Visualization
# =============================================================================


def stable_colors(count: int, seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    return [
        (
            rng.randint(50, 245),
            rng.randint(50, 245),
            rng.randint(50, 245),
        )
        for _ in range(count)
    ]


def selected_masks_overlay(
    image: Image.Image,
    record: dict,
    selected_indices: np.ndarray,
    seed: int,
    alpha: float = 0.60,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    colors = stable_colors(len(selected_indices), seed)

    for order_index, pred_index in enumerate(selected_indices):
        prediction = record["predictions"][int(pred_index)]
        mask = decode_rle(prediction["rle"])
        color_array = np.asarray(colors[order_index], dtype=np.float32)
        base[mask] = (1.0 - alpha) * base[mask] + alpha * color_array

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def gt_recovered_overlay(
    image: Image.Image,
    record: dict,
    recovered_indices: set[int],
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)

    for gt_index in sorted(recovered_indices):
        mask = decode_rle(record["gt_rles"][gt_index])
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255), width=3)
        draw.text((x0 + 2, max(0, y0 - 14)), f"RECOVERED GT #{gt_index}", fill=(255, 255, 255))

    return result


def save_recovery_examples(
    project_root: Path,
    baseline_records: list[dict],
    enhanced_records: list[dict],
    baseline_config: dict,
    enhanced_config: dict,
    baseline_matches: dict[str, set[int]],
    enhanced_matches: dict[str, set[int]],
    output_dir: Path,
    max_examples: int,
    seed: int,
) -> list[str]:
    baseline_lookup = {record["sample_id"]: record for record in baseline_records}
    enhanced_lookup = {record["sample_id"]: record for record in enhanced_records}

    ranked: list[tuple[int, str]] = []
    for sample_id in sorted(enhanced_lookup):
        recovered = enhanced_matches.get(sample_id, set()) - baseline_matches.get(sample_id, set())
        if recovered:
            ranked.append((len(recovered), sample_id))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    ranked = ranked[:max_examples]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for example_index, (_, sample_id) in enumerate(ranked, start=1):
        baseline_record = baseline_lookup[sample_id]
        enhanced_record = enhanced_lookup[sample_id]

        image_path = resolve_row_path(project_root, enhanced_record["image_path"])
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        baseline_selected = select_prediction_indices(
            baseline_record,
            full_confidence=baseline_config["full_confidence"],
            tile_confidence=baseline_config["tile_confidence"],
            min_area_pixels=baseline_config["min_area_pixels"],
            max_area_fraction=baseline_config["max_area_fraction"],
            nms_iou=baseline_config["mask_nms_iou"],
        )

        enhanced_selected = select_prediction_indices(
            enhanced_record,
            full_confidence=enhanced_config["full_confidence"],
            tile_confidence=enhanced_config["tile_confidence"],
            min_area_pixels=enhanced_config["min_area_pixels"],
            max_area_fraction=enhanced_config["max_area_fraction"],
            nms_iou=enhanced_config["mask_nms_iou"],
        )

        recovered = enhanced_matches.get(sample_id, set()) - baseline_matches.get(sample_id, set())

        baseline_panel = selected_masks_overlay(
            image, baseline_record, baseline_selected, seed=seed + example_index
        )
        enhanced_panel = selected_masks_overlay(
            image, enhanced_record, enhanced_selected, seed=seed + 1000 + example_index
        )
        recovered_panel = gt_recovered_overlay(image, enhanced_record, recovered)

        width, height = image.size
        header = 64
        canvas = Image.new("RGB", (width * 4, height + header), "white")
        canvas.paste(image, (0, header))
        canvas.paste(baseline_panel, (width, header))
        canvas.paste(enhanced_panel, (width * 2, header))
        canvas.paste(recovered_panel, (width * 3, header))

        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"Sample: {sample_id} | recovered GT={len(recovered)}", fill="black")
        draw.text((10, 36), "ORIGINAL", fill="black")
        draw.text((width + 10, 36), "FULL OPTIMIZED", fill="black")
        draw.text((width * 2 + 10, 36), enhanced_config["strategy"], fill="black")
        draw.text((width * 3 + 10, 36), "RECOVERED GT BOXES", fill="black")

        output_path = output_dir / f"recovery_{example_index:02d}_{sample_id}.jpg"
        canvas.save(output_path, quality=92)
        saved.append(relative_or_absolute(output_path, project_root))

    return saved


def save_strategy_plot(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    names = [row["strategy"] for row in rows]
    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - 1.5 * width, [row["precision"] for row in rows], width, label="Precision")
    ax.bar(x - 0.5 * width, [row["recall"] for row in rows], width, label="Recall")
    ax.bar(x + 0.5 * width, [row["f1"] for row in rows], width, label="F1")
    ax.bar(x + 1.5 * width, [row["f2"] for row in rows], width, label="F2")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("SAM 3 Occlusion / Multi-Scale Strategy Comparison")
    ax.grid(axis="y", alpha=0.25)
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
        description=(
            "Compare full-image SAM 3 with overlapping tiled and occlusion-aware "
            "multi-prompt inference on the frozen MinneApple validation split."
        )
    )

    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--subset-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV containing a sample_id column. Useful later for a manually "
            "curated occlusion-heavy image subset."
        ),
    )

    parser.add_argument("--full-prompt", type=str, default=DEFAULT_FULL_PROMPT)
    parser.add_argument("--occlusion-prompts", nargs="*", default=None)
    parser.add_argument("--base-confidence", type=float, default=DEFAULT_BASE_CONFIDENCE)
    parser.add_argument("--full-confidence", type=float, default=DEFAULT_FULL_CONFIDENCE)
    parser.add_argument("--tile-thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--tile-min-areas", nargs="*", type=int, default=None)
    parser.add_argument("--max-area-fraction", type=float, default=DEFAULT_MAX_AREA_FRACTION)
    parser.add_argument("--nms-ious", nargs="*", type=float, default=None)

    parser.add_argument("--grid-rows", type=int, default=DEFAULT_GRID_ROWS)
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS)
    parser.add_argument("--tile-overlap", type=float, default=DEFAULT_TILE_OVERLAP)

    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--accept-ious", nargs="*", type=float, default=None)
    parser.add_argument(
        "--objective",
        choices=["f1", "f2", "review_actions"],
        default=DEFAULT_OBJECTIVE,
        help="Default f2 emphasizes recall for occlusion recovery.",
    )

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--recovery-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "The official MinneApple test split is frozen. Use validation for this "
            "experiment. Pass --allow-test only if you explicitly intend to break "
            "the development protocol."
        )

    if not 0.0 <= args.base_confidence <= 1.0:
        raise SystemExit("--base-confidence must be between 0 and 1.")
    if not 0.0 <= args.full_confidence <= 1.0:
        raise SystemExit("--full-confidence must be between 0 and 1.")
    if args.full_confidence < args.base_confidence:
        raise SystemExit("--full-confidence cannot be below --base-confidence.")
    if not 0.0 < args.max_area_fraction <= 1.0:
        raise SystemExit("--max-area-fraction must be in (0, 1].")
    if not 0.0 <= args.tile_overlap < 1.0:
        raise SystemExit("--tile-overlap must be in [0, 1).")
    if args.grid_rows <= 0 or args.grid_cols <= 0:
        raise SystemExit("--grid-rows and --grid-cols must be > 0.")
    if not 0.0 <= args.match_iou <= 1.0:
        raise SystemExit("--match-iou must be between 0 and 1.")

    occlusion_prompts = args.occlusion_prompts or list(DEFAULT_OCCLUSION_PROMPTS)
    occlusion_prompts = [prompt.strip() for prompt in occlusion_prompts if prompt.strip()]
    occlusion_prompts = list(dict.fromkeys(occlusion_prompts))
    if args.full_prompt not in occlusion_prompts:
        occlusion_prompts.insert(0, args.full_prompt)

    tile_thresholds = parse_unique_floats(
        args.tile_thresholds,
        DEFAULT_TILE_THRESHOLDS,
        "--tile-thresholds",
        args.base_confidence,
        1.0,
    )
    tile_min_areas = parse_unique_ints(
        args.tile_min_areas,
        DEFAULT_TILE_MIN_AREAS,
        "--tile-min-areas",
        minimum=0,
    )
    nms_ious = parse_unique_floats(
        args.nms_ious,
        DEFAULT_FUSION_NMS_IOUS,
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

    project_root = resolve_project_root(args.project_root)
    manifest_path = resolve_manifest_path(project_root, args.manifest)

    subset_path = None
    if args.subset_csv:
        subset_path = Path(args.subset_csv).expanduser()
        if not subset_path.is_absolute():
            subset_path = project_root / subset_path
        subset_path = subset_path.resolve()

    experiment_name = f"sam3_occlusion_multiscale_{args.split}"
    if args.limit is not None:
        experiment_name += f"_limit{args.limit}"
    if subset_path is not None:
        experiment_name += "_subset"

    experiment_root = (project_root / OUTPUTS_RELATIVE_PATH / experiment_name).resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)

    summary_path = experiment_root / "occlusion_multiscale_summary.json"
    cache_index_csv = experiment_root / "cache_index.csv"
    strategy_search_csv = experiment_root / "strategy_search.csv"
    strategy_comparison_csv = experiment_root / "strategy_comparison.csv"
    per_image_csv = experiment_root / "selected_per_image_metrics.csv"
    area_recall_csv = experiment_root / "visible_area_stratified_recall.csv"
    recovery_csv = experiment_root / "recovered_gt_instances.csv"
    comparison_plot = experiment_root / "strategy_comparison.png"
    recovery_examples_dir = experiment_root / "recovery_examples"

    summary: dict = {
        "experiment_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "status": "RUNNING",
        "project_root": str(project_root),
        "manifest": relative_or_absolute(manifest_path, project_root),
        "split": args.split,
        "subset_csv": relative_or_absolute(subset_path, project_root) if subset_path else None,
        "task_definition": "visible instance segmentation (not amodal segmentation)",
        "full_prompt": args.full_prompt,
        "occlusion_prompts": occlusion_prompts,
        "base_confidence": args.base_confidence,
        "full_confidence": args.full_confidence,
        "tile_thresholds": tile_thresholds,
        "tile_min_areas": tile_min_areas,
        "max_area_fraction": args.max_area_fraction,
        "fusion_nms_ious": nms_ious,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "tile_overlap": args.tile_overlap,
        "matching_iou": args.match_iou,
        "auto_accept_ious": accept_ious,
        "selection_objective": args.objective,
        "errors": [],
        "warnings": [],
    }

    print()
    print("=" * 104)
    print("SAM 3 OCCLUSION / MULTI-SCALE EXPERIMENT — MINNEAPPLE")
    print("=" * 104)
    print(f"Project root           : {project_root}")
    print(f"Manifest               : {manifest_path}")
    print(f"Split                  : {args.split}")
    print(f"Full prompt            : {args.full_prompt!r}")
    print(f"Occlusion prompts      : {occlusion_prompts}")
    print(f"Grid                   : {args.grid_rows} x {args.grid_cols}")
    print(f"Tile overlap           : {args.tile_overlap:.2f}")
    print(f"Base confidence        : {args.base_confidence:.2f}")
    print(f"Full confidence        : {args.full_confidence:.2f}")
    print(f"Tile thresholds        : {tile_thresholds}")
    print(f"Tile min areas         : {tile_min_areas}")
    print(f"Max area fraction      : {args.max_area_fraction}")
    print(f"Fusion NMS IoUs        : {nms_ious}")
    print(f"Selection objective    : {args.objective}")
    print(f"Output                 : {experiment_root}")
    print()

    try:
        # ------------------------------------------------------------------
        # 1/8 Environment + data
        # ------------------------------------------------------------------
        print("[1/8] Auditing environment and loading validation data...")

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
                "CPU inference is disabled by default. Use CUDA or pass --allow-cpu."
            )

        summary["device"] = device

        rows = load_manifest(manifest_path)
        subset_ids = load_subset_ids(subset_path)
        selected_rows = select_split_rows(rows, args.split, args.limit, subset_ids)
        total_manifest_gt = sum(
            int(float(row.get("instance_count", "0") or 0)) for row in selected_rows
        )

        summary["selected_images"] = len(selected_rows)
        summary["selected_gt_instances_manifest"] = int(total_manifest_gt)

        print(f"  Device                : {device}")
        print(f"  Selected images       : {len(selected_rows)}")
        print(f"  Manifest GT apples    : {total_manifest_gt}")

        # ------------------------------------------------------------------
        # 2/8 Inspect caches / determine missing inference
        # ------------------------------------------------------------------
        print()
        print("[2/8] Inspecting full-image and tile caches...")

        full_cache_lookup: dict[str, Path] = {}
        tile_cache_lookup: dict[tuple[str, str], Path] = {}
        pending_full: list[dict[str, str]] = []
        pending_tiles: list[tuple[str, dict[str, str]]] = []

        for row in selected_rows:
            sample_id = row["sample_id"]

            full_path = None
            full_source = "missing"
            if not args.overwrite:
                full_path, full_source = load_compatible_full_cache(
                    project_root,
                    experiment_root,
                    args.split,
                    row,
                    args.full_prompt,
                    args.base_confidence,
                )

            if full_path is None:
                pending_full.append(row)
            else:
                full_cache_lookup[sample_id] = full_path

            for prompt in occlusion_prompts:
                tile_path = None
                if not args.overwrite:
                    tile_path = load_compatible_tile_cache(
                        experiment_root,
                        row,
                        prompt,
                        args.base_confidence,
                        args.grid_rows,
                        args.grid_cols,
                        args.tile_overlap,
                    )

                if tile_path is None:
                    pending_tiles.append((prompt, row))
                else:
                    tile_cache_lookup[(prompt, sample_id)] = tile_path

        print(f"  Missing full-image jobs: {len(pending_full)}")
        print(f"  Missing tile/prompt jobs: {len(pending_tiles)}")
        print(
            f"  Tile SAM calls if uncached: {len(pending_tiles) * args.grid_rows * args.grid_cols}"
        )

        # ------------------------------------------------------------------
        # 3/8 Run missing SAM 3 inference
        # ------------------------------------------------------------------
        print()
        print("[3/8] Running missing SAM 3 inference...")

        package_info = None
        checkpoint_info = None
        model_load_seconds = 0.0
        peak_memory_bytes = None
        full_inference_times: list[float] = []
        tile_job_times: list[float] = []

        if pending_full or pending_tiles:
            build_sam3_image_model, Sam3Processor, package_info = import_sam3(project_root)
            checkpoint_path = resolve_checkpoint(project_root, args.checkpoint)

            if checkpoint_path is None and not args.allow_hf_download:
                expected = project_root / DEFAULT_CHECKPOINT_RELATIVE_PATH
                raise RuntimeError(
                    "No local SAM 3 checkpoint was found.\n"
                    f"Expected: {expected}\n"
                    "Provide --checkpoint or pass --allow-hf-download after Hugging Face login."
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

            use_autocast = (
                device == "cuda"
                and not args.no_autocast
                and bool(environment.get("bf16_supported", False))
            )

            print(f"  Model load time       : {model_load_seconds:.3f}s")

            # Full-image jobs, usually reused from Stage 4B.
            for index, row in enumerate(pending_full, start=1):
                sample_id = row["sample_id"]
                image_path = resolve_row_path(project_root, row["processed_image_path"])

                with Image.open(image_path) as image_file:
                    image = image_file.convert("RGB")

                masks, boxes, scores, seconds, autocast_label = run_sam3_inference_one(
                    processor,
                    image,
                    args.full_prompt,
                    device,
                    use_autocast,
                )

                predictions = []
                for pred_index in range(len(masks)):
                    predictions.append(
                        {
                            "instance_index": pred_index + 1,
                            "class_label": "apple",
                            "prompt": args.full_prompt,
                            "source": "full",
                            "score": float(scores[pred_index]),
                            "box_xyxy": [float(value) for value in boxes[pred_index][:4]],
                            "mask_area_pixels": int(masks[pred_index].sum()),
                            "rle": encode_binary_mask(masks[pred_index]),
                        }
                    )

                output_path = full_local_cache_path(
                    experiment_root, args.full_prompt, sample_id
                )
                save_full_cache(
                    output_path,
                    full_cache_signature(row, args.full_prompt, args.base_confidence),
                    row,
                    image.size,
                    predictions,
                    seconds,
                    autocast_label,
                )
                full_cache_lookup[sample_id] = output_path
                full_inference_times.append(seconds)

                if index % 20 == 0 or index == len(pending_full):
                    print(
                        f"  Full inference        : {index}/{len(pending_full)} | "
                        f"preds={len(predictions)} | {seconds:.3f}s"
                    )

            # Tile-prompt jobs.
            total_tile_jobs = len(pending_tiles)
            for job_index, (prompt, row) in enumerate(pending_tiles, start=1):
                sample_id = row["sample_id"]
                image_path = resolve_row_path(project_root, row["processed_image_path"])

                with Image.open(image_path) as image_file:
                    image = image_file.convert("RGB")

                full_width, full_height = image.size
                tiles = generate_overlapping_tiles(
                    full_width,
                    full_height,
                    args.grid_rows,
                    args.grid_cols,
                    args.tile_overlap,
                )

                all_predictions: list[dict] = []
                total_seconds = 0.0
                autocast_label = "unknown"

                for tile in tiles:
                    crop = image.crop((tile["x0"], tile["y0"], tile["x1"], tile["y1"]))
                    masks, boxes, scores, seconds, autocast_label = run_sam3_inference_one(
                        processor,
                        crop,
                        prompt,
                        device,
                        use_autocast,
                    )
                    total_seconds += seconds
                    all_predictions.extend(
                        map_tile_predictions_to_full(
                            masks,
                            boxes,
                            scores,
                            tile,
                            full_width,
                            full_height,
                            prompt,
                        )
                    )

                output_path = tile_cache_path(experiment_root, prompt, sample_id)
                save_tile_cache(
                    output_path,
                    tile_cache_signature(
                        row,
                        prompt,
                        args.base_confidence,
                        args.grid_rows,
                        args.grid_cols,
                        args.tile_overlap,
                    ),
                    row,
                    image.size,
                    tiles,
                    all_predictions,
                    total_seconds,
                    autocast_label,
                )
                tile_cache_lookup[(prompt, sample_id)] = output_path
                tile_job_times.append(total_seconds)

                if job_index % 20 == 0 or job_index == total_tile_jobs:
                    print(
                        f"  Tile/prompt progress  : {job_index}/{total_tile_jobs} | "
                        f"prompt={prompt!r} | preds={len(all_predictions)} | "
                        f"tiles={len(tiles)} | {total_seconds:.3f}s"
                    )

            if device == "cuda":
                peak_memory_bytes = int(torch.cuda.max_memory_allocated())

            del processor
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        else:
            print("  All required inference is already cached.")

        summary["sam3_package"] = package_info
        summary["checkpoint"] = checkpoint_info
        summary["new_inference"] = {
            "full_image_jobs": len(full_inference_times),
            "tile_prompt_jobs": len(tile_job_times),
            "mean_full_seconds": float(np.mean(full_inference_times))
            if full_inference_times
            else 0.0,
            "mean_tile_prompt_seconds": float(np.mean(tile_job_times))
            if tile_job_times
            else 0.0,
            "model_load_seconds": model_load_seconds,
            "cuda_peak_memory_bytes": peak_memory_bytes,
            "cuda_peak_memory": human_bytes(peak_memory_bytes),
        }

        # Refresh all cache lookups and write cache index.
        cache_rows: list[dict] = []
        for row in selected_rows:
            sample_id = row["sample_id"]
            full_path, full_source = load_compatible_full_cache(
                project_root,
                experiment_root,
                args.split,
                row,
                args.full_prompt,
                args.base_confidence,
            )
            if full_path is None:
                raise RuntimeError(f"Missing full cache after inference: {sample_id}")
            full_cache_lookup[sample_id] = full_path
            cache_rows.append(
                {
                    "sample_id": sample_id,
                    "kind": "full",
                    "prompt": args.full_prompt,
                    "cache_source": full_source,
                    "cache_path": relative_or_absolute(full_path, project_root),
                }
            )

            for prompt in occlusion_prompts:
                path = load_compatible_tile_cache(
                    experiment_root,
                    row,
                    prompt,
                    args.base_confidence,
                    args.grid_rows,
                    args.grid_cols,
                    args.tile_overlap,
                )
                if path is None:
                    raise RuntimeError(
                        f"Missing tile cache after inference: sample={sample_id}, prompt={prompt!r}"
                    )
                tile_cache_lookup[(prompt, sample_id)] = path
                cache_rows.append(
                    {
                        "sample_id": sample_id,
                        "kind": "tile",
                        "prompt": prompt,
                        "cache_source": "local",
                        "cache_path": relative_or_absolute(path, project_root),
                    }
                )

        write_csv(
            cache_index_csv,
            cache_rows,
            ["sample_id", "kind", "prompt", "cache_source", "cache_path"],
        )

        # ------------------------------------------------------------------
        # 4/8 Build three evaluation record sets
        # ------------------------------------------------------------------
        print()
        print("[4/8] Building full / tiled / multi-prompt evaluation records...")

        records_full: list[dict] = []
        records_tiles: list[dict] = []
        records_multi: list[dict] = []

        for index, row in enumerate(selected_rows, start=1):
            sample_id = row["sample_id"]
            full_cache = full_cache_lookup[sample_id]

            # A: full image only.
            records_full.append(
                load_strategy_record(
                    project_root,
                    row,
                    full_cache,
                    tile_caches=[],
                )
            )

            # B: full + tiles with only the established orchard prompt.
            records_tiles.append(
                load_strategy_record(
                    project_root,
                    row,
                    full_cache,
                    tile_caches=[
                        (args.full_prompt, tile_cache_lookup[(args.full_prompt, sample_id)])
                    ],
                )
            )

            # C: full + tiles from all occlusion-aware prompts.
            records_multi.append(
                load_strategy_record(
                    project_root,
                    row,
                    full_cache,
                    tile_caches=[
                        (prompt, tile_cache_lookup[(prompt, sample_id)])
                        for prompt in occlusion_prompts
                    ],
                )
            )

            if index % 20 == 0 or index == len(selected_rows):
                print(
                    f"  Record progress       : {index}/{len(selected_rows)} | "
                    f"full={records_full[-1]['pred_count']} | "
                    f"tiles={records_tiles[-1]['pred_count']} | "
                    f"multi={records_multi[-1]['pred_count']}"
                )

        # ------------------------------------------------------------------
        # 5/8 Evaluate current full-image baseline
        # ------------------------------------------------------------------
        print()
        print("[5/8] Evaluating frozen FULL_OPTIMIZED baseline...")

        baseline_metrics, baseline_per_image, baseline_matches = evaluate_records(
            records_full,
            strategy_name="FULL_OPTIMIZED",
            full_confidence=args.full_confidence,
            tile_confidence=args.full_confidence,
            min_area_pixels=DEFAULT_MIN_AREA_PIXELS,
            max_area_fraction=args.max_area_fraction,
            nms_iou=DEFAULT_MASK_NMS_IOU,
            match_iou=args.match_iou,
            accept_ious=accept_ious,
            include_per_image=True,
        )

        print(
            f"  FULL_OPTIMIZED        : P={baseline_metrics['precision']:.4f} "
            f"R={baseline_metrics['recall']:.4f} F1={baseline_metrics['f1']:.4f} "
            f"F2={baseline_metrics['f2']:.4f} mIoU={baseline_metrics['mean_matched_iou']:.4f}"
        )

        # ------------------------------------------------------------------
        # 6/8 Search tiled fusion configurations
        # ------------------------------------------------------------------
        print()
        print("[6/8] Searching tiled and occlusion-aware fusion configurations...")

        tile_search_rows = search_enhanced_strategy(
            records_tiles,
            strategy_name="FULL_PLUS_TILES",
            full_confidence=args.full_confidence,
            tile_thresholds=tile_thresholds,
            min_areas=tile_min_areas,
            max_area_fraction=args.max_area_fraction,
            nms_ious=nms_ious,
            match_iou=args.match_iou,
            accept_ious=accept_ious,
        )

        multi_search_rows = search_enhanced_strategy(
            records_multi,
            strategy_name="FULL_PLUS_TILES_MULTI_PROMPT",
            full_confidence=args.full_confidence,
            tile_thresholds=tile_thresholds,
            min_areas=tile_min_areas,
            max_area_fraction=args.max_area_fraction,
            nms_ious=nms_ious,
            match_iou=args.match_iou,
            accept_ious=accept_ious,
        )

        all_search_rows = tile_search_rows + multi_search_rows
        write_csv(
            strategy_search_csv,
            all_search_rows,
            list(all_search_rows[0].keys()) if all_search_rows else [],
        )

        tile_best_f1 = best_row(tile_search_rows, "f1")
        tile_best_f2 = best_row(tile_search_rows, "f2")
        tile_best_review = best_row(tile_search_rows, "review_actions")
        multi_best_f1 = best_row(multi_search_rows, "f1")
        multi_best_f2 = best_row(multi_search_rows, "f2")
        multi_best_review = best_row(multi_search_rows, "review_actions")

        # The selected configuration follows the requested objective and can
        # still fall back to FULL_OPTIMIZED if the enhanced strategies are worse.
        objective_candidates = [
            baseline_metrics,
            best_row(tile_search_rows, args.objective),
            best_row(multi_search_rows, args.objective),
        ]
        selected_config = max(
            objective_candidates,
            key=lambda row: config_rank_key(row, args.objective),
        )

        if selected_config["strategy"] == "FULL_OPTIMIZED":
            selected_records = records_full
        elif selected_config["strategy"] == "FULL_PLUS_TILES":
            selected_records = records_tiles
        else:
            selected_records = records_multi

        selected_metrics, selected_per_image, selected_matches = evaluate_records(
            selected_records,
            strategy_name=selected_config["strategy"],
            full_confidence=selected_config["full_confidence"],
            tile_confidence=selected_config["tile_confidence"],
            min_area_pixels=selected_config["min_area_pixels"],
            max_area_fraction=selected_config["max_area_fraction"],
            nms_iou=selected_config["mask_nms_iou"],
            match_iou=args.match_iou,
            accept_ious=accept_ious,
            include_per_image=True,
        )

        # Evaluate the two enhanced strategy representatives under the chosen
        # objective so the final table always has three directly comparable rows.
        tile_selected_config = best_row(tile_search_rows, args.objective)
        tile_selected_metrics, _, tile_selected_matches = evaluate_records(
            records_tiles,
            strategy_name="FULL_PLUS_TILES",
            full_confidence=tile_selected_config["full_confidence"],
            tile_confidence=tile_selected_config["tile_confidence"],
            min_area_pixels=tile_selected_config["min_area_pixels"],
            max_area_fraction=tile_selected_config["max_area_fraction"],
            nms_iou=tile_selected_config["mask_nms_iou"],
            match_iou=args.match_iou,
            accept_ious=accept_ious,
            include_per_image=False,
        )

        multi_selected_config = best_row(multi_search_rows, args.objective)
        multi_selected_metrics, _, multi_selected_matches = evaluate_records(
            records_multi,
            strategy_name="FULL_PLUS_TILES_MULTI_PROMPT",
            full_confidence=multi_selected_config["full_confidence"],
            tile_confidence=multi_selected_config["tile_confidence"],
            min_area_pixels=multi_selected_config["min_area_pixels"],
            max_area_fraction=multi_selected_config["max_area_fraction"],
            nms_iou=multi_selected_config["mask_nms_iou"],
            match_iou=args.match_iou,
            accept_ious=accept_ious,
            include_per_image=False,
        )

        comparison_rows = [baseline_metrics, tile_selected_metrics, multi_selected_metrics]
        write_csv(
            strategy_comparison_csv,
            comparison_rows,
            list(comparison_rows[0].keys()),
        )
        save_strategy_plot(comparison_plot, comparison_rows)

        print()
        print("  Strategy comparison under objective =", args.objective)
        for row in comparison_rows:
            print(
                f"    {row['strategy']:<30} "
                f"P={row['precision']:.4f} R={row['recall']:.4f} "
                f"F1={row['f1']:.4f} F2={row['f2']:.4f} "
                f"mIoU={row['mean_matched_iou']:.4f}"
            )

        # ------------------------------------------------------------------
        # 7/8 Area-stratified recall + recovered GT analysis
        # ------------------------------------------------------------------
        print()
        print("[7/8] Analyzing small-visible / recovered GT instances...")

        # Use FULL records for GT areas; GT is identical across strategies.
        area_edges = compute_gt_area_quartile_edges(records_full)

        area_rows: list[dict] = []
        area_rows.extend(
            stratified_recall_rows(
                records_full,
                baseline_matches,
                "FULL_OPTIMIZED",
                area_edges,
            )
        )
        area_rows.extend(
            stratified_recall_rows(
                records_tiles,
                tile_selected_matches,
                "FULL_PLUS_TILES",
                area_edges,
            )
        )
        area_rows.extend(
            stratified_recall_rows(
                records_multi,
                multi_selected_matches,
                "FULL_PLUS_TILES_MULTI_PROMPT",
                area_edges,
            )
        )

        write_csv(
            area_recall_csv,
            area_rows,
            ["strategy", "visible_area_bin", "gt_instances", "matched_instances", "recall"],
        )

        recovery_detail_rows, recovery_summary = recovery_rows(
            records=selected_records,
            baseline_matches=baseline_matches,
            enhanced_matches=selected_matches,
            enhanced_strategy=selected_metrics["strategy"],
            edges=area_edges,
        )

        write_csv(
            recovery_csv,
            recovery_detail_rows,
            [
                "sample_id",
                "change",
                "gt_index",
                "gt_visible_area_pixels",
                "visible_area_bin",
                "enhanced_strategy",
            ],
        )

        # Per-image output from selected strategy.
        write_csv(
            per_image_csv,
            selected_per_image,
            list(selected_per_image[0].keys()) if selected_per_image else [],
        )

        print(
            f"  Visible GT area quartiles: Q1={area_edges[0]:.1f}, "
            f"Q2={area_edges[1]:.1f}, Q3={area_edges[2]:.1f} pixels"
        )
        print(
            f"  Recovered GT          : {recovery_summary['recovered_gt_instances']} | "
            f"Lost GT: {recovery_summary['lost_gt_instances']} | "
            f"Net: {recovery_summary['net_gt_recovery']:+d}"
        )

        # ------------------------------------------------------------------
        # 8/8 Recovery visualizations + summary
        # ------------------------------------------------------------------
        print()
        print("[8/8] Saving recovery visualizations and final summary...")

        baseline_visual_config = {
            "strategy": "FULL_OPTIMIZED",
            "full_confidence": args.full_confidence,
            "tile_confidence": args.full_confidence,
            "min_area_pixels": DEFAULT_MIN_AREA_PIXELS,
            "max_area_fraction": args.max_area_fraction,
            "mask_nms_iou": DEFAULT_MASK_NMS_IOU,
        }
        enhanced_visual_config = {
            "strategy": selected_metrics["strategy"],
            "full_confidence": selected_metrics["full_confidence"],
            "tile_confidence": selected_metrics["tile_confidence"],
            "min_area_pixels": selected_metrics["min_area_pixels"],
            "max_area_fraction": selected_metrics["max_area_fraction"],
            "mask_nms_iou": selected_metrics["mask_nms_iou"],
        }

        recovery_examples = save_recovery_examples(
            project_root=project_root,
            baseline_records=records_full,
            enhanced_records=selected_records,
            baseline_config=baseline_visual_config,
            enhanced_config=enhanced_visual_config,
            baseline_matches=baseline_matches,
            enhanced_matches=selected_matches,
            output_dir=recovery_examples_dir,
            max_examples=max(0, args.recovery_examples),
            seed=args.seed,
        )

        summary["strategy_a_full_optimized"] = baseline_metrics
        summary["strategy_b_full_plus_tiles"] = {
            "best_by_f1": tile_best_f1,
            "best_by_f2": tile_best_f2,
            "best_by_review_actions": tile_best_review,
            "selected_for_objective": tile_selected_metrics,
        }
        summary["strategy_c_full_plus_tiles_multi_prompt"] = {
            "best_by_f1": multi_best_f1,
            "best_by_f2": multi_best_f2,
            "best_by_review_actions": multi_best_review,
            "selected_for_objective": multi_selected_metrics,
        }
        summary["selected_configuration"] = {
            "objective": args.objective,
            "strategy": selected_metrics["strategy"],
            "full_confidence": selected_metrics["full_confidence"],
            "tile_confidence": selected_metrics["tile_confidence"],
            "min_area_pixels": selected_metrics["min_area_pixels"],
            "max_area_fraction": selected_metrics["max_area_fraction"],
            "mask_nms_iou": selected_metrics["mask_nms_iou"],
        }
        summary["selected_metrics"] = selected_metrics
        summary["delta_vs_full_optimized"] = {
            "precision": selected_metrics["precision"] - baseline_metrics["precision"],
            "recall": selected_metrics["recall"] - baseline_metrics["recall"],
            "f1": selected_metrics["f1"] - baseline_metrics["f1"],
            "f2": selected_metrics["f2"] - baseline_metrics["f2"],
            "mean_matched_iou": selected_metrics["mean_matched_iou"]
            - baseline_metrics["mean_matched_iou"],
            "false_negatives": selected_metrics["fn"] - baseline_metrics["fn"],
            "false_positives": selected_metrics["fp"] - baseline_metrics["fp"],
        }
        summary["visible_gt_area_quartiles_pixels"] = {
            "q1": area_edges[0],
            "q2": area_edges[1],
            "q3": area_edges[2],
            "note": (
                "Small visible area is only a proxy for difficult/partially visible apples; "
                "MinneApple does not provide an explicit occlusion severity label here."
            ),
        }
        summary["recovery_vs_full_optimized"] = recovery_summary
        summary["recovery_examples"] = recovery_examples
        summary["outputs"] = {
            "experiment_root": relative_or_absolute(experiment_root, project_root),
            "cache_index_csv": relative_or_absolute(cache_index_csv, project_root),
            "strategy_search_csv": relative_or_absolute(strategy_search_csv, project_root),
            "strategy_comparison_csv": relative_or_absolute(
                strategy_comparison_csv, project_root
            ),
            "selected_per_image_csv": relative_or_absolute(per_image_csv, project_root),
            "visible_area_stratified_recall_csv": relative_or_absolute(
                area_recall_csv, project_root
            ),
            "recovered_gt_csv": relative_or_absolute(recovery_csv, project_root),
            "strategy_plot": relative_or_absolute(comparison_plot, project_root),
            "summary_json": relative_or_absolute(summary_path, project_root),
        }

        summary["status"] = "PASS_WITH_WARNINGS" if summary["warnings"] else "PASS"
        write_json(summary_path, summary)

        print()
        print("=" * 104)
        print("SAM 3 OCCLUSION / MULTI-SCALE EXPERIMENT COMPLETE")
        print("=" * 104)
        print(f"STATUS                        : {summary['status']}")
        print(f"IMAGES                        : {len(selected_rows)}")
        print(f"GT APPLE INSTANCES            : {total_manifest_gt}")
        print(f"SELECTION OBJECTIVE           : {args.objective}")
        print(f"SELECTED STRATEGY             : {selected_metrics['strategy']}")
        print(f"FULL CONFIDENCE               : {selected_metrics['full_confidence']:.2f}")
        print(f"TILE CONFIDENCE               : {selected_metrics['tile_confidence']:.2f}")
        print(f"MIN MASK AREA                 : {selected_metrics['min_area_pixels']}")
        print(f"FUSION MASK NMS               : {selected_metrics['mask_nms_iou']:.2f}")
        print(f"PRECISION                     : {selected_metrics['precision']:.4f}")
        print(f"RECALL                        : {selected_metrics['recall']:.4f}")
        print(f"F1                            : {selected_metrics['f1']:.4f}")
        print(f"F2                            : {selected_metrics['f2']:.4f}")
        print(f"MEAN MATCHED IoU              : {selected_metrics['mean_matched_iou']:.4f}")
        print(f"DELTA RECALL VS FULL          : {summary['delta_vs_full_optimized']['recall']:+.4f}")
        print(f"DELTA F1 VS FULL              : {summary['delta_vs_full_optimized']['f1']:+.4f}")
        print(f"DELTA F2 VS FULL              : {summary['delta_vs_full_optimized']['f2']:+.4f}")
        print(
            f"RECOVERED / LOST / NET GT     : "
            f"{recovery_summary['recovered_gt_instances']} / "
            f"{recovery_summary['lost_gt_instances']} / "
            f"{recovery_summary['net_gt_recovery']:+d}"
        )
        print(f"SUMMARY                       : {summary_path}")
        print(f"STRATEGY COMPARISON           : {strategy_comparison_csv}")
        print(f"AREA-STRATIFIED RECALL        : {area_recall_csv}")
        print(f"RECOVERY EXAMPLES             : {recovery_examples_dir}")
        print("=" * 104)

    except Exception as error:
        summary["status"] = "FAIL"
        summary["errors"].append(str(error))
        summary["exception_type"] = type(error).__name__
        summary["traceback"] = traceback.format_exc()
        write_json(summary_path, summary)

        print()
        print("=" * 104)
        print("SAM 3 OCCLUSION / MULTI-SCALE EXPERIMENT FAILED")
        print("=" * 104)
        print(f"ERROR   : {error}")
        print(f"REPORT  : {summary_path}")
        print("=" * 104)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
