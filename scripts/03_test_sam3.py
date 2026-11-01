from __future__ import annotations

from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from importlib import metadata as importlib_metadata

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw


# =============================================================================
# SAM 3 Smoke Test for AgriAnnotationSystem
# =============================================================================
#
# Purpose
# -------
# 1. Verify Python / PyTorch / CUDA / GPU environment.
# 2. Verify the official SAM 3 package can be imported.
# 3. Load the official SAM 3 image model.
# 4. Select ONE deterministic MinneApple validation image from the frozen
#    data/splits/minneapple_split.csv manifest.
# 5. Run text-prompt inference with prompt="apple".
# 6. Validate masks / boxes / scores returned by SAM 3.
# 7. Save a 3-panel visualization:
#       original | ground truth | SAM 3 prediction
# 8. Save prediction metadata and a structured JSON smoke-test report.
#
# This script is NOT a benchmark. It does not compute dataset-wide IoU/AP.
# Full zero-shot evaluation belongs in 04_run_zero_shot_baseline.py.
#
# Official SAM 3 image API used here:
#   from sam3.model_builder import build_sam3_image_model
#   from sam3.model.sam3_image_processor import Sam3Processor
#
# Recommended official prerequisites (as of 2026):
#   Python >= 3.12
#   PyTorch >= 2.7
#   CUDA >= 12.6
#
# =============================================================================


MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)
MIN_CUDA = (12, 6)

DEFAULT_PROMPT = "apple"
DEFAULT_CONFIDENCE = 0.50

MANIFEST_RELATIVE_PATH = Path("data/splits/minneapple_split.csv")
DEFAULT_CHECKPOINT_RELATIVE_PATH = Path("models/sam3/checkpoints/sam3.pt")
DEFAULT_REPORT_RELATIVE_PATH = Path("outputs/reports/sam3_smoke_test")
THIRD_PARTY_SAM3_RELATIVE_PATH = Path("third_party/sam3")


# =============================================================================
# Basic utilities
# =============================================================================


def version_tuple(text: str | None, length: int = 3) -> tuple[int, ...]:
    """Extract a comparable numeric version tuple from strings like 2.10.0+cu128."""
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def resolve_project_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    # Expected location:
    # AgriAnnotationSystem/scripts/03_test_sam3.py
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


# =============================================================================
# Environment audit
# =============================================================================


def collect_environment() -> tuple[dict, list[str], list[str]]:
    """Return (environment, errors, warnings)."""
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
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is recommended by the official "
            f"SAM 3 repository; found {platform.python_version()}."
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
                    f"the PyTorch build reports CUDA {torch.version.cuda}."
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
            warnings.append("PyTorch cannot currently access a CUDA GPU.")

    except Exception as error:
        environment["torch_import_error"] = repr(error)
        errors.append(f"PyTorch import failed: {error}")

    return environment, errors, warnings


# =============================================================================
# Manifest handling
# =============================================================================


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen MinneApple manifest not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    required = {
        "sample_id",
        "split",
        "processed_image_path",
        "processed_mask_path",
        "instance_count",
    }

    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")

    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(
            "Manifest is missing required columns: " + ", ".join(sorted(missing))
        )

    return rows


def choose_sample(
    rows: list[dict[str, str]],
    split: str,
    sample_id: str | None,
) -> dict[str, str]:
    candidates = [row for row in rows if row.get("split", "").strip() == split]

    if not candidates:
        raise RuntimeError(f"No samples found for split='{split}'.")

    if sample_id:
        matches = [row for row in candidates if row.get("sample_id") == sample_id]
        if not matches:
            raise RuntimeError(
                f"sample_id='{sample_id}' was not found in split='{split}'."
            )
        return matches[0]

    # Representative deterministic sample:
    # choose the sample whose GT instance count is closest to the split median.
    def count(row: dict[str, str]) -> int:
        try:
            return int(float(row.get("instance_count", "0")))
        except Exception:
            return 0

    sorted_rows = sorted(
        candidates,
        key=lambda row: (count(row), row.get("sample_id", "")),
    )

    return sorted_rows[len(sorted_rows) // 2]


# =============================================================================
# SAM 3 import and checkpoint resolution
# =============================================================================


def add_local_sam3_to_path(project_root: Path) -> Path | None:
    repo = (project_root / THIRD_PARTY_SAM3_RELATIVE_PATH).resolve()

    # A cloned official repo normally contains:
    # third_party/sam3/sam3/model_builder.py
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
        guidance = (
            "SAM 3 could not be imported. Install the official repository in the active "
            "Python environment, for example:\n\n"
            "  cd <SAM3_REPOSITORY>\n"
            "  python -m pip install -e .\n\n"
            "This script also recognizes a cloned repository at:\n"
            f"  {project_root / THIRD_PARTY_SAM3_RELATIVE_PATH}\n"
        )
        raise RuntimeError(guidance) from error

    package_info = {
        "sam3_version": getattr(sam3, "__version__", safe_package_version("sam3")),
        "sam3_module_file": getattr(sam3, "__file__", None),
        "local_repo_detected": str(local_repo) if local_repo else None,
    }

    return build_sam3_image_model, Sam3Processor, package_info


def resolve_checkpoint(
    project_root: Path,
    explicit_checkpoint: str | None,
) -> Path | None:
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
        # Prefer official SAM 3 image checkpoint if a user renamed or copied it.
        candidates = sorted(checkpoint_dir.glob("*.pt"))
        sam3_candidates = [
            p
            for p in candidates
            if "sam3" in p.name.lower() and "3.1" not in p.name.lower()
        ]

        if len(sam3_candidates) == 1:
            return sam3_candidates[0].resolve()

    return None


# =============================================================================
# Ground truth + prediction helpers
# =============================================================================


def load_gt_mask(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {path}")

    with Image.open(path) as image:
        mask = np.array(image)

    if mask.ndim != 2:
        raise RuntimeError(
            f"Expected a 2-D indexed MinneApple mask, found shape={mask.shape}"
        )

    ids = np.unique(mask)
    instance_ids = ids[ids != 0]
    return mask, int(len(instance_ids))


def _torch_tensor_to_numpy(value) -> np.ndarray:
    """Safely move a PyTorch tensor to CPU and convert it to NumPy.

    SAM 3 keeps prediction tensors on the inference device. When CUDA BF16
    autocast is enabled, boxes/scores may also be bfloat16. NumPy does not
    reliably accept PyTorch bfloat16 tensors, so floating dtypes that NumPy
    cannot represent directly are promoted to float32 before conversion.
    """
    import torch

    tensor = value.detach()

    # NumPy conversion of bfloat16 is not supported consistently.
    # Preserve boolean/integer tensors, but promote BF16/FP8-style values.
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(dtype=torch.float32)
    elif str(tensor.dtype).startswith("torch.float8"):
        tensor = tensor.to(dtype=torch.float32)

    tensor = tensor.to(device="cpu")
    return tensor.numpy()


def tensor_to_numpy(value) -> np.ndarray:
    """Convert tensors / tensor sequences / array-like values to NumPy safely."""
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


def normalize_prediction_masks(masks, expected_height: int, expected_width: int) -> np.ndarray:
    """Convert SAM 3 masks to bool [N, H, W] on CPU."""
    if masks is None:
        return np.zeros((0, expected_height, expected_width), dtype=bool)

    masks = tensor_to_numpy(masks)

    if masks.size == 0:
        return np.zeros((0, expected_height, expected_width), dtype=bool)

    # Official Sam3Processor currently returns [N, 1, H, W].
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
# Visualization
# =============================================================================


def stable_colors(count: int, seed: int) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    return [
        (
            rng.randint(40, 245),
            rng.randint(40, 245),
            rng.randint(40, 245),
        )
        for _ in range(count)
    ]


def draw_text_tag(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    accent: tuple[int, int, int],
) -> None:
    """Draw a readable annotation tag with a dark background."""
    x, y = xy
    try:
        left, top, right, bottom = draw.textbbox((x, y), text)
    except Exception:
        left, top, right, bottom = x, y, x + max(40, len(text) * 7), y + 12

    pad_x = 3
    pad_y = 2
    draw.rectangle(
        (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y),
        fill=(0, 0, 0),
        outline=accent,
        width=1,
    )
    draw.text((x, y), text, fill=(255, 255, 255))


def compute_mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a thin boolean contour for a binary mask."""
    mask = mask.astype(bool, copy=False)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbors = [
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[:-2, :-2],
        padded[:-2, 2:],
        padded[2:, :-2],
        padded[2:, 2:],
    ]
    interior = center.copy()
    for neighbor in neighbors:
        interior &= neighbor
    boundary = center & (~interior)
    return boundary


def overlay_indexed_gt(image: Image.Image, gt_mask: np.ndarray, seed: int) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    ids = np.unique(gt_mask)
    ids = ids[ids != 0]
    colors = stable_colors(len(ids), seed)
    alpha = 0.62

    for instance_id, color in zip(ids, colors):
        region = gt_mask == instance_id
        color_array = np.asarray(color, dtype=np.float32)
        base[region] = (1.0 - alpha) * base[region] + alpha * color_array
        boundary = compute_mask_boundary(region)
        base[boundary] = color_array

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def overlay_prediction_masks_only(
    image: Image.Image,
    masks: np.ndarray,
    scores: np.ndarray,
    class_label: str,
    seed: int,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    colors = stable_colors(len(masks), seed)
    alpha = 0.68

    for index, mask in enumerate(masks):
        color = colors[index] if index < len(colors) else (255, 255, 255)
        color_array = np.asarray(color, dtype=np.float32)
        region = mask.astype(bool, copy=False)
        base[region] = (1.0 - alpha) * base[region] + alpha * color_array
        boundary = compute_mask_boundary(region)
        base[boundary] = color_array

    result = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)

    # Label only the top scoring predictions to keep the mask panel readable.
    if len(scores) > 0:
        top_k = min(12, len(scores))
        ranked = np.argsort(scores)[::-1][:top_k]
        for rank_index in ranked:
            ys, xs = np.where(masks[rank_index])
            if len(xs) == 0 or len(ys) == 0:
                continue
            x = int(np.median(xs))
            y = int(np.median(ys))
            color = colors[rank_index] if rank_index < len(colors) else (255, 255, 255)
            tag = f"{class_label} #{rank_index + 1}"
            draw_text_tag(draw, (x + 2, max(0, y - 12)), tag, color)

    return result


def overlay_prediction_boxes_only(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_label: str,
    seed: int,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    colors = stable_colors(len(boxes), seed)

    for index, box in enumerate(boxes):
        if len(box) < 4:
            continue

        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        color = colors[index] if index < len(colors) else (255, 255, 255)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)

        score_text = ""
        if index < len(scores):
            score_text = f" {float(scores[index]):.3f}"
        tag = f"{class_label} #{index + 1}{score_text}"
        draw_text_tag(draw, (int(x0) + 2, max(0, int(y0) - 14)), tag, color)

    return result


def make_comparison_canvas(
    original: Image.Image,
    gt_overlay: Image.Image,
    pred_mask_overlay: Image.Image,
    pred_box_overlay: Image.Image,
    sample_id: str,
    prompt: str,
    gt_count: int,
    pred_count: int,
) -> Image.Image:
    width, height = original.size
    header = 58

    canvas = Image.new("RGB", (width * 4, height + header), "white")
    canvas.paste(original, (0, header))
    canvas.paste(gt_overlay, (width, header))
    canvas.paste(pred_mask_overlay, (width * 2, header))
    canvas.paste(pred_box_overlay, (width * 3, header))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"Sample: {sample_id} | prompt='{prompt}'", fill="black")
    draw.text((10, 34), "ORIGINAL", fill="black")
    draw.text((width + 10, 34), f"GROUND TRUTH MASKS | apples={gt_count}", fill="black")
    draw.text((width * 2 + 10, 34), f"SAM 3 MASKS ONLY | instances={pred_count}", fill="black")
    draw.text((width * 3 + 10, 34), f"SAM 3 BOXES + SCORES | instances={pred_count}", fill="black")

    return canvas


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the official SAM 3 image model on one frozen MinneApple validation sample."
    )

    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help=(
            "If no local models/sam3/checkpoints/sam3.pt exists, allow the official "
            "SAM 3 builder to download the gated checkpoint from Hugging Face. "
            "You must already have checkpoint access and be authenticated."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU inference. This is not recommended for SAM 3.",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable CUDA bfloat16 autocast during inference.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save each predicted binary mask as an individual PNG file.",
    )
    parser.add_argument(
        "--allow-test-sample",
        action="store_true",
        help="Explicitly permit a smoke test on the frozen official test split.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1.")

    project_root = resolve_project_root(args.project_root)
    manifest_path = resolve_manifest_path(project_root, args.manifest)
    report_root = (project_root / DEFAULT_REPORT_RELATIVE_PATH).resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    summary_path = report_root / "smoke_test_summary.json"
    metadata_csv = report_root / "prediction_metadata.csv"
    visualization_path = report_root / "sam3_smoke_test_visualization.jpg"
    mask_output_dir = report_root / "prediction_masks"

    summary: dict = {
        "smoke_test_version": "1.3",
        "created_at": datetime.now().isoformat(),
        "status": "RUNNING",
        "project_root": str(project_root),
        "manifest": relative_or_absolute(manifest_path, project_root),
        "prompt": args.prompt,
        "confidence_threshold": args.confidence,
        "requested_split": args.split,
        "errors": [],
        "warnings": [],
    }

    print()
    print("=" * 80)
    print("SAM 3 / MINNEAPPLE SMOKE TEST")
    print("=" * 80)
    print(f"Project root : {project_root}")
    print(f"Manifest     : {manifest_path}")
    print(f"Prompt       : {args.prompt!r}")
    print(f"Confidence   : {args.confidence:.3f}")
    print()

    try:
        # ---------------------------------------------------------------------
        # 1/7 Environment
        # ---------------------------------------------------------------------
        print("[1/7] Auditing Python / PyTorch / CUDA environment...")
        environment, env_errors, env_warnings = collect_environment()
        summary["environment"] = environment
        summary["errors"].extend(env_errors)
        summary["warnings"].extend(env_warnings)

        for key in [
            "python_version",
            "python_executable",
            "torch_version",
            "torchvision_version",
            "torch_cuda_version",
            "cuda_available",
            "cuda_device_name",
            "cuda_compute_capability",
            "cuda_total_memory",
            "bf16_supported",
        ]:
            if key in environment:
                print(f"  {key:<25}: {environment[key]}")

        # Import torch only after the structured audit.
        import torch

        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but PyTorch reports CUDA unavailable.")

        if device == "cpu" and not args.allow_cpu:
            raise RuntimeError(
                "CUDA is unavailable and CPU execution is disabled by default. "
                "Fix the CUDA/PyTorch environment or rerun with --allow-cpu for a slow diagnostic run."
            )

        summary["device"] = device
        print(f"  Selected device          : {device}")

        if env_errors:
            raise RuntimeError(
                "Environment does not meet the SAM 3 prerequisites:\n  - "
                + "\n  - ".join(env_errors)
            )

        # ---------------------------------------------------------------------
        # 2/7 Manifest + sample
        # ---------------------------------------------------------------------
        print()
        print("[2/7] Loading frozen MinneApple manifest...")

        if args.split == "test" and not args.allow_test_sample:
            raise RuntimeError(
                "The official MinneApple test split is frozen. "
                "Use the validation split for smoke testing. "
                "If you intentionally need a test image, pass --allow-test-sample."
            )

        rows = load_manifest(manifest_path)
        sample = choose_sample(rows, args.split, args.sample_id)

        image_path_value = sample.get("processed_image_path") or sample.get("image_path", "")
        mask_path_value = sample.get("processed_mask_path") or sample.get("mask_path", "")

        if not image_path_value or not mask_path_value:
            raise RuntimeError("Selected manifest row does not contain usable image/mask paths.")

        image_path = resolve_row_path(project_root, image_path_value)
        gt_mask_path = resolve_row_path(project_root, mask_path_value)

        if not image_path.is_file():
            raise FileNotFoundError(f"Processed image not found: {image_path}")

        gt_mask, gt_count_from_mask = load_gt_mask(gt_mask_path)

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        width, height = image.size

        manifest_gt_count = int(float(sample.get("instance_count", "0") or 0))
        if manifest_gt_count != gt_count_from_mask:
            summary["warnings"].append(
                f"Manifest instance_count={manifest_gt_count} but mask contains "
                f"{gt_count_from_mask} non-background instance IDs."
            )

        sample_info = {
            "sample_id": sample.get("sample_id"),
            "split": sample.get("split"),
            "group_id": sample.get("group_id"),
            "image_path": relative_or_absolute(image_path, project_root),
            "mask_path": relative_or_absolute(gt_mask_path, project_root),
            "image_width": width,
            "image_height": height,
            "ground_truth_instances": gt_count_from_mask,
        }
        summary["sample"] = sample_info

        print(f"  Sample ID                : {sample_info['sample_id']}")
        print(f"  Split                    : {sample_info['split']}")
        print(f"  Group                    : {sample_info['group_id']}")
        print(f"  Image                    : {image_path}")
        print(f"  Resolution               : {width}x{height}")
        print(f"  GT apple instances       : {gt_count_from_mask}")

        # ---------------------------------------------------------------------
        # 3/7 Import SAM 3
        # ---------------------------------------------------------------------
        print()
        print("[3/7] Importing official SAM 3 package...")

        build_sam3_image_model, Sam3Processor, package_info = import_sam3(project_root)
        summary["sam3_package"] = package_info

        print(f"  SAM 3 version            : {package_info['sam3_version']}")
        print(f"  SAM 3 module             : {package_info['sam3_module_file']}")
        if package_info["local_repo_detected"]:
            print(f"  Local repository         : {package_info['local_repo_detected']}")

        # ---------------------------------------------------------------------
        # 4/7 Resolve checkpoint
        # ---------------------------------------------------------------------
        print()
        print("[4/7] Resolving SAM 3 checkpoint...")

        checkpoint_path = resolve_checkpoint(project_root, args.checkpoint)

        if checkpoint_path is None and not args.allow_hf_download:
            expected = project_root / DEFAULT_CHECKPOINT_RELATIVE_PATH
            raise RuntimeError(
                "No local official SAM 3 image checkpoint was found.\n"
                f"Expected location:\n  {expected}\n\n"
                "Either place the official sam3.pt there, pass --checkpoint <path>, "
                "or rerun with --allow-hf-download after obtaining Hugging Face "
                "checkpoint access and authenticating with `hf auth login`."
            )

        if checkpoint_path is not None:
            checkpoint_source = "local"
            print(f"  Checkpoint               : {checkpoint_path}")
            print(f"  Size                     : {human_bytes(checkpoint_path.stat().st_size)}")
        else:
            checkpoint_source = "huggingface_auto"
            print("  Checkpoint               : Hugging Face automatic download/cache")

        summary["checkpoint"] = {
            "source": checkpoint_source,
            "path": str(checkpoint_path) if checkpoint_path else None,
            "allow_hf_download": bool(args.allow_hf_download),
        }

        # ---------------------------------------------------------------------
        # 5/7 Load model
        # ---------------------------------------------------------------------
        print()
        print("[5/7] Loading SAM 3 image model...")

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
            confidence_threshold=args.confidence,
        )

        if device == "cuda":
            torch.cuda.synchronize()

        load_seconds = time.perf_counter() - load_start
        print(f"  Model load time          : {load_seconds:.3f} s")

        # ---------------------------------------------------------------------
        # 6/7 Run one image
        # ---------------------------------------------------------------------
        print()
        print("[6/7] Running text-prompt inference...")

        use_autocast = (
            device == "cuda"
            and not args.no_autocast
            and bool(environment.get("bf16_supported", False))
        )

        if use_autocast:
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            autocast_label = "cuda/bfloat16"
        else:
            autocast_context = nullcontext()
            autocast_label = "disabled"

        print(f"  Autocast                 : {autocast_label}")

        if device == "cuda":
            torch.cuda.synchronize()

        inference_start = time.perf_counter()

        with torch.inference_mode():
            with autocast_context:
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=args.prompt)

        if device == "cuda":
            torch.cuda.synchronize()

        inference_seconds = time.perf_counter() - inference_start

        if not isinstance(output, dict):
            raise RuntimeError(f"SAM 3 returned unexpected output type: {type(output)}")

        masks = normalize_prediction_masks(output.get("masks"), height, width)
        boxes = tensor_to_numpy(output.get("boxes"))
        scores = tensor_to_numpy(output.get("scores")).reshape(-1)

        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
        else:
            boxes = np.asarray(boxes).reshape(-1, 4)

        pred_count = int(len(masks))

        if len(boxes) != pred_count:
            raise RuntimeError(
                f"Prediction count mismatch: masks={pred_count}, boxes={len(boxes)}"
            )

        if len(scores) != pred_count:
            raise RuntimeError(
                f"Prediction count mismatch: masks={pred_count}, scores={len(scores)}"
            )

        peak_memory_bytes = None
        if device == "cuda":
            peak_memory_bytes = int(torch.cuda.max_memory_allocated())

        count_ratio = (
            float(pred_count / gt_count_from_mask)
            if gt_count_from_mask > 0
            else None
        )

        inference_info = {
            "prompt": args.prompt,
            "confidence_threshold": args.confidence,
            "autocast": autocast_label,
            "model_load_seconds": round(load_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "predicted_instances": pred_count,
            "ground_truth_instances": gt_count_from_mask,
            "prediction_to_gt_count_ratio": round(count_ratio, 6)
            if count_ratio is not None
            else None,
            "mask_shape": list(masks.shape),
            "boxes_shape": list(boxes.shape),
            "scores_shape": list(scores.shape),
            "score_min": float(scores.min()) if pred_count else None,
            "score_mean": float(scores.mean()) if pred_count else None,
            "score_max": float(scores.max()) if pred_count else None,
            "cuda_peak_memory_bytes": peak_memory_bytes,
            "cuda_peak_memory": human_bytes(peak_memory_bytes),
        }
        summary["inference"] = inference_info

        print(f"  Inference time           : {inference_seconds:.3f} s")
        print(f"  Predicted instances      : {pred_count}")
        print(f"  GT instances             : {gt_count_from_mask}")
        print(f"  Count ratio pred/GT      : {count_ratio:.3f}" if count_ratio is not None else "  Count ratio pred/GT      : N/A")
        print(f"  Mask shape               : {masks.shape}")
        print(f"  Box shape                : {boxes.shape}")
        print(f"  Score shape              : {scores.shape}")

        if pred_count > 0:
            print(
                "  Score min/mean/max       : "
                f"{scores.min():.4f} / {scores.mean():.4f} / {scores.max():.4f}"
            )

        if peak_memory_bytes is not None:
            print(f"  CUDA peak allocated      : {human_bytes(peak_memory_bytes)}")

        if pred_count == 0:
            summary["warnings"].append(
                "SAM 3 inference completed successfully but returned zero instances for prompt='"
                + args.prompt
                + "'. Keep the singular noun 'apple' for the formal baseline and investigate prompt/threshold only on the validation split."
            )

        # ---------------------------------------------------------------------
        # 7/7 Save outputs
        # ---------------------------------------------------------------------
        print()
        print("[7/7] Saving smoke-test outputs...")

        prediction_rows: list[dict] = []

        for index in range(pred_count):
            box = boxes[index]
            mask_area = int(masks[index].sum())

            prediction_rows.append(
                {
                    "instance_index": index + 1,
                    "class_label": args.prompt,
                    "score": float(scores[index]),
                    "x0": float(box[0]),
                    "y0": float(box[1]),
                    "x1": float(box[2]),
                    "y1": float(box[3]),
                    "mask_area_pixels": mask_area,
                }
            )

        write_csv(
            metadata_csv,
            prediction_rows,
            [
                "instance_index",
                "class_label",
                "score",
                "x0",
                "y0",
                "x1",
                "y1",
                "mask_area_pixels",
            ],
        )

        if args.save_masks:
            mask_output_dir.mkdir(parents=True, exist_ok=True)

            for index, mask in enumerate(masks, start=1):
                mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
                mask_image.save(mask_output_dir / f"mask_{index:04d}.png")

        gt_overlay = overlay_indexed_gt(image, gt_mask, seed=args.seed)
        pred_mask_overlay = overlay_prediction_masks_only(
            image,
            masks,
            scores,
            class_label=args.prompt,
            seed=args.seed + 1000,
        )
        pred_box_overlay = overlay_prediction_boxes_only(
            image,
            boxes,
            scores,
            class_label=args.prompt,
            seed=args.seed + 2000,
        )

        canvas = make_comparison_canvas(
            image,
            gt_overlay,
            pred_mask_overlay,
            pred_box_overlay,
            sample_id=sample_info["sample_id"],
            prompt=args.prompt,
            gt_count=gt_count_from_mask,
            pred_count=pred_count,
        )
        canvas.save(visualization_path, quality=92)

        summary["outputs"] = {
            "summary_json": relative_or_absolute(summary_path, project_root),
            "prediction_metadata_csv": relative_or_absolute(metadata_csv, project_root),
            "visualization": relative_or_absolute(visualization_path, project_root),
            "prediction_masks_directory": (
                relative_or_absolute(mask_output_dir, project_root)
                if args.save_masks
                else None
            ),
        }

        summary["status"] = "PASS_WITH_WARNINGS" if summary["warnings"] else "PASS"
        write_json(summary_path, summary)

        print(f"  Visualization           : {visualization_path}")
        print(f"  Prediction metadata     : {metadata_csv}")
        print(f"  Summary                 : {summary_path}")

        print()
        print("=" * 80)
        print("SAM 3 SMOKE TEST COMPLETE")
        print("=" * 80)
        print(f"STATUS                   : {summary['status']}")
        print(f"DEVICE                   : {device}")
        print(f"SAMPLE                   : {sample_info['sample_id']}")
        print(f"PROMPT                   : {args.prompt}")
        print(f"GROUND TRUTH INSTANCES   : {gt_count_from_mask}")
        print(f"PREDICTED INSTANCES      : {pred_count}")
        print(f"INFERENCE TIME           : {inference_seconds:.3f} s")
        if peak_memory_bytes is not None:
            print(f"CUDA PEAK MEMORY         : {human_bytes(peak_memory_bytes)}")
        print(f"VISUALIZATION            : {visualization_path}")
        print(f"SUMMARY                  : {summary_path}")
        print("=" * 80)

        # Explicit cleanup after report generation.
        try:
            del output
            del state
            del processor
            del model
        except Exception:
            pass

        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as error:
        summary["status"] = "FAIL"
        summary["errors"].append(str(error))
        summary["exception_type"] = type(error).__name__
        summary["traceback"] = traceback.format_exc()
        write_json(summary_path, summary)

        print()
        print("=" * 80)
        print("SAM 3 SMOKE TEST FAILED")
        print("=" * 80)
        print(f"ERROR   : {error}")
        print(f"REPORT  : {summary_path}")
        print("=" * 80)

        raise SystemExit(1)


if __name__ == "__main__":
    main()
