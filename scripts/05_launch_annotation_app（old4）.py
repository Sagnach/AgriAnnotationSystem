from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import argparse
import hashlib
import json
import math
import sys
import threading
import time
import uuid

import numpy as np
import gradio as gr  # module scope required for event type-hint resolution in Gradio 6
from PIL import Image, ImageDraw, ImageOps


# =============================================================================
# AgriAnnotationSystem - SAM 3 Human-in-the-loop Annotation UI
# =============================================================================
#
# MVP capabilities
# ----------------
# 1. Upload one agricultural image.
# 2. Run SAM 3 text-prompt instance segmentation in two inference modes:
#       FAST      : full-image optimized inference (~0.21 s/image on RTX 5080)
#       DEEP SCAN : full image + 2x2 overlapping tiles for better small/occluded recall
# 3. Apply frozen MinneApple validation settings from Stage 4B / Stage 4C:
#       Fast primary confidence       ~0.45
#       Fast candidate confidence     ~0.30
#       Deep primary confidence       ~0.45
#       Deep candidate confidence     ~0.40
#       Deep grid                     2 x 2
#       Deep overlap                  20%
#       min area                      200 px
#       max area                      1% of full image
#       Fast mask NMS                 0.70
#       Deep fusion mask NMS          0.60
#    Values are automatically loaded from the Stage 4B / Stage 4C summaries
#    when available.
# 4. Display class label + instance ID + score.
# 5. Human review:
#       - select instance by clicking its mask
#       - accept
#       - delete
#       - relabel
#       - add a missed object using a positive click
#       - refine a selected mask with positive / negative clicks
#       - show / hide low-confidence candidates
# 6. Save reviewed annotations and export one-image COCO Instance Segmentation.
# 7. Add another class pass (for future apple/orange/pear multi-class use).
# 8. B1 interactive image canvas:
#       - click rendered label chip and edit class inline
# 9. B2 interactive bounding box editing:
#       - drag selected rectangle, resize eight handles
#       - Cancel discards draft; Apply uses a SAM 3 box-only prompt
#       - commit the new instance mask + derived bbox atomically
#       - update review snapshot and COCO export
#       - retain original image point tools
#
# The UI intentionally keeps model code, state management and persistence
# separate. Annotation state is owned by src/annotation/annotation_manager.py.
# =============================================================================


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------


def resolve_project_root(explicit_root: str | None = None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = resolve_project_root(None)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.annotation_manager import AnnotationManager, bbox_from_mask  # noqa: E402
from src.ui.interactive_canvas import (  # noqa: E402
    CANVAS_HTML, CANVAS_CSS, CANVAS_JS, empty_canvas_payload, make_canvas_payload,
)


DEFAULT_OPTIMIZATION_SUMMARY = Path(
    "outputs/experiments/sam3_zero_shot_optimization_val/optimization_summary.json"
)
DEFAULT_OCCLUSION_SUMMARY = Path(
    "outputs/experiments/sam3_occlusion_multiscale_val/occlusion_multiscale_summary.json"
)
DEFAULT_CHECKPOINT = Path("models/sam3/checkpoints/sam3.pt")
THIRD_PARTY_SAM3 = Path("third_party/sam3")


# =============================================================================
# Configuration
# =============================================================================


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_optimized_defaults(project_root: Path) -> dict[str, Any]:
    defaults = {
        "prompt": "orchard apple",
        "class_label": "apple",
        "primary_confidence": 0.45,
        "candidate_confidence": 0.30,
        "min_area_pixels": 200,
        "max_area_fraction": 0.01,
        "mask_nms_iou": 0.70,
        "source": "built_in_fallback",
    }

    summary_path = project_root / DEFAULT_OPTIMIZATION_SUMMARY
    if not summary_path.is_file():
        return defaults

    try:
        summary = read_json(summary_path)
        selected = summary.get("selected_configuration", {})
        best_f2 = summary.get("stage_b_postprocess_search", {}).get("best_by_f2", {})

        primary = float(selected.get("confidence_threshold", defaults["primary_confidence"]))
        candidate = float(best_f2.get("confidence_threshold", defaults["candidate_confidence"]))
        candidate = min(candidate, primary)

        defaults.update(
            {
                "prompt": selected.get("prompt", defaults["prompt"]),
                "primary_confidence": primary,
                "candidate_confidence": candidate,
                "min_area_pixels": int(
                    selected.get("min_area_pixels", defaults["min_area_pixels"])
                ),
                "max_area_fraction": float(
                    selected.get("max_area_fraction", defaults["max_area_fraction"])
                ),
                "mask_nms_iou": float(
                    selected.get("mask_nms_iou", defaults["mask_nms_iou"])
                ),
                "source": str(summary_path.relative_to(project_root)),
            }
        )
    except Exception:
        # The UI should remain launchable even if a prior experiment summary is
        # missing or malformed. The fallback values are the frozen v1 settings.
        pass

    return defaults


def load_deep_scan_defaults(
    project_root: Path,
    fast_defaults: dict[str, Any],
) -> dict[str, Any]:
    """Load the frozen single-prompt tiled configuration from Stage 4C.

    We intentionally use FULL_PLUS_TILES rather than the much slower multi-prompt
    strategy because Stage 4C showed almost identical recall/F2 at roughly one
    third of the inference time. The best-F1 tiled configuration is used for the
    primary layer, while the best-F2 tile threshold becomes the optional
    low-confidence candidate floor.
    """
    defaults = {
        "prompt": str(fast_defaults.get("prompt", "orchard apple")),
        "primary_confidence": 0.45,
        "candidate_confidence": 0.40,
        "min_area_pixels": 200,
        "max_area_fraction": 0.01,
        "fusion_mask_nms_iou": 0.60,
        "grid_rows": 2,
        "grid_cols": 2,
        "tile_overlap": 0.20,
        "source": "built_in_fallback",
    }

    summary_path = project_root / DEFAULT_OCCLUSION_SUMMARY
    if not summary_path.is_file():
        return defaults

    try:
        summary = read_json(summary_path)
        tiled = summary.get("strategy_b_full_plus_tiles", {})
        best_f1 = tiled.get("best_by_f1", {})
        best_f2 = tiled.get("best_by_f2", {})

        defaults.update(
            {
                "prompt": str(summary.get("full_prompt", defaults["prompt"])),
                "primary_confidence": float(
                    best_f1.get("tile_confidence", defaults["primary_confidence"])
                ),
                "candidate_confidence": min(
                    float(best_f2.get("tile_confidence", defaults["candidate_confidence"])),
                    float(best_f1.get("tile_confidence", defaults["primary_confidence"])),
                ),
                "min_area_pixels": int(
                    best_f1.get("min_area_pixels", defaults["min_area_pixels"])
                ),
                "max_area_fraction": float(
                    best_f1.get("max_area_fraction", defaults["max_area_fraction"])
                ),
                "fusion_mask_nms_iou": float(
                    best_f1.get("mask_nms_iou", defaults["fusion_mask_nms_iou"])
                ),
                "grid_rows": int(summary.get("grid_rows", defaults["grid_rows"])),
                "grid_cols": int(summary.get("grid_cols", defaults["grid_cols"])),
                "tile_overlap": float(summary.get("tile_overlap", defaults["tile_overlap"])),
                "source": str(summary_path.relative_to(project_root)),
            }
        )
    except Exception:
        pass

    return defaults


def generate_overlapping_tiles(
    width: int,
    height: int,
    rows: int,
    cols: int,
    overlap_fraction: float,
) -> list[dict[str, int]]:
    """Generate the same overlapping grid used in Stage 4C."""
    if rows <= 0 or cols <= 0:
        raise ValueError("Tile rows/cols must be > 0.")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("Tile overlap must be in [0, 1).")

    cell_w = width / cols
    cell_h = height / rows
    expand_x = cell_w * overlap_fraction / 2.0
    expand_y = cell_h * overlap_fraction / 2.0

    tiles: list[dict[str, int]] = []
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
                }
            )
            tile_index += 1

    return tiles


def deep_mask_nms_indices(
    masks: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Global mask NMS matching Stage 4C, with a tiny full-image tie bonus."""
    if len(masks) == 0:
        return np.empty((0,), dtype=np.int64)

    source_bonus = np.where(np.asarray(sources) == "full", 1e-6, 0.0)
    order = np.argsort(np.asarray(scores, dtype=np.float32) + source_bonus)[::-1]
    if iou_threshold <= 0.0:
        return order.astype(np.int64)

    from pycocotools import mask as mask_utils

    rles = encode_masks_for_iou(masks)
    pairwise = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float32)
    selected: list[int] = []
    for pred_index in order:
        pred_index = int(pred_index)
        if selected:
            overlaps = pairwise[pred_index, selected]
            if bool(np.any(overlaps >= float(iou_threshold))):
                continue
        selected.append(pred_index)
    return np.asarray(selected, dtype=np.int64)


# =============================================================================
# SAM 3 runtime
# =============================================================================


def add_local_sam3_to_path(project_root: Path) -> Path | None:
    repo = (project_root / THIRD_PARTY_SAM3).resolve()
    if (repo / "sam3" / "model_builder.py").is_file():
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        return repo
    return None


def resolve_checkpoint(project_root: Path, explicit_checkpoint: str | None) -> Path | None:
    if explicit_checkpoint:
        path = Path(explicit_checkpoint).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SAM 3 checkpoint not found: {path}")
        return path

    preferred = (project_root / DEFAULT_CHECKPOINT).resolve()
    if preferred.is_file():
        return preferred

    checkpoint_dir = preferred.parent
    if checkpoint_dir.is_dir():
        candidates = [
            path
            for path in checkpoint_dir.glob("*.pt")
            if "sam3" in path.name.lower() and "3.1" not in path.name.lower()
        ]
        if len(candidates) == 1:
            return candidates[0].resolve()

    return None


def _torch_tensor_to_numpy(value) -> np.ndarray:
    import torch

    tensor = value.detach()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(dtype=torch.float32)
    elif str(tensor.dtype).startswith("torch.float8"):
        tensor = tensor.to(dtype=torch.float32)
    return tensor.to(device="cpu").numpy()


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
            f"SAM 3 mask size {masks.shape[1:]} does not match image "
            f"{(expected_height, expected_width)}."
        )

    return masks.astype(bool, copy=False)


def encode_masks_for_iou(masks: np.ndarray) -> list[dict[str, Any]]:
    from pycocotools import mask as mask_utils

    result: list[dict[str, Any]] = []
    for mask in masks:
        result.append(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
    return result


def mask_nms_indices(masks: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Score-ordered mask NMS using pycocotools IoU."""
    if len(masks) == 0:
        return np.empty((0,), dtype=np.int64)
    if iou_threshold <= 0:
        return np.arange(len(masks), dtype=np.int64)

    from pycocotools import mask as mask_utils

    order = np.argsort(scores)[::-1]
    rles = encode_masks_for_iou(masks)
    pairwise = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=np.float32)

    keep: list[int] = []
    suppressed = np.zeros(len(masks), dtype=bool)

    for index in order:
        if suppressed[index]:
            continue
        keep.append(int(index))
        overlaps = pairwise[index] > float(iou_threshold)
        overlaps[index] = False
        suppressed |= overlaps

    return np.asarray(keep, dtype=np.int64)


class Sam3Runtime:
    def __init__(
        self,
        project_root: Path,
        checkpoint: str | None = None,
        allow_hf_download: bool = True,
        device: str = "auto",
    ) -> None:
        self.project_root = project_root.resolve()
        self.explicit_checkpoint = checkpoint
        self.allow_hf_download = bool(allow_hf_download)
        self.requested_device = device

        self.model = None
        self.processor = None
        self.device = None
        self.checkpoint_source = None
        self.local_repo = None
        self._lock = threading.RLock()

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self, candidate_confidence: float = 0.30) -> str:
        with self._lock:
            if self.is_loaded():
                self.processor.confidence_threshold = float(candidate_confidence)
                return self.status_text()

            import torch

            self.local_repo = add_local_sam3_to_path(self.project_root)

            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except Exception as error:
                raise RuntimeError(
                    "SAM 3 could not be imported. Confirm that the official repository "
                    "is installed in the active .venv312 environment."
                ) from error

            if self.requested_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.requested_device

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")

            checkpoint = resolve_checkpoint(self.project_root, self.explicit_checkpoint)
            load_from_hf = checkpoint is None and self.allow_hf_download

            if checkpoint is None and not load_from_hf:
                raise RuntimeError(
                    "No local SAM 3 checkpoint was found and Hugging Face download is disabled."
                )

            kwargs = {
                "device": device,
                "eval_mode": True,
                "enable_segmentation": True,
                # Needed for click-to-add and click-based refinement.
                "enable_inst_interactivity": True,
                "load_from_HF": load_from_hf,
            }
            if checkpoint is not None:
                kwargs["checkpoint_path"] = str(checkpoint)

            self.model = build_sam3_image_model(**kwargs)
            self.processor = Sam3Processor(
                self.model,
                device=device,
                confidence_threshold=float(candidate_confidence),
            )
            self.device = device
            self.checkpoint_source = str(checkpoint) if checkpoint is not None else "huggingface_auto"

            if getattr(self.model, "inst_interactive_predictor", None) is None:
                raise RuntimeError(
                    "SAM 3 loaded without instance interactivity. The UI requires "
                    "enable_inst_interactivity=True for Add/Refine click tools."
                )

            return self.status_text()

    def status_text(self) -> str:
        if not self.is_loaded():
            return "SAM 3 model: not loaded"

        import torch

        gpu = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        return (
            f"SAM 3 model: loaded | device={self.device} | {gpu} | "
            f"checkpoint={self.checkpoint_source}"
        )

    def _autocast_context(self):
        import torch

        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _infer_text_raw(
        self,
        image: Image.Image,
        prompt: str,
        confidence_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Run one SAM 3 text pass without structural filtering/NMS."""
        import torch

        self.load(candidate_confidence=confidence_threshold)
        self.processor.confidence_threshold = float(confidence_threshold)

        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size

        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            with self._autocast_context():
                state = self.processor.set_image(image)
                output = self.processor.set_text_prompt(state=state, prompt=str(prompt))

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        masks = normalize_prediction_masks(output.get("masks"), height, width)
        boxes = tensor_to_numpy(output.get("boxes"))
        scores = tensor_to_numpy(output.get("scores")).reshape(-1)

        boxes = (
            np.empty((0, 4), dtype=np.float32)
            if boxes.size == 0
            else np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if len(masks) != len(boxes) or len(masks) != len(scores):
            raise RuntimeError(
                f"SAM 3 count mismatch: masks={len(masks)}, boxes={len(boxes)}, scores={len(scores)}"
            )

        return masks, boxes, scores, elapsed

    def infer_text(
        self,
        image: Image.Image,
        prompt: str,
        candidate_confidence: float,
        min_area_pixels: int,
        max_area_fraction: float,
        mask_nms_iou: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast full-image inference used by the original UI mode."""
        with self._lock:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            masks, boxes, scores, _elapsed = self._infer_text_raw(
                image=image,
                prompt=prompt,
                confidence_threshold=float(candidate_confidence),
            )

            if len(masks) == 0:
                return masks, boxes, scores

            areas = masks.reshape(len(masks), -1).sum(axis=1)
            max_area = float(max_area_fraction) * float(width * height)
            keep = (
                (scores >= float(candidate_confidence))
                & (areas >= int(min_area_pixels))
                & (areas <= max_area)
            )
            masks = masks[keep]
            boxes = boxes[keep]
            scores = scores[keep]

            keep_indices = mask_nms_indices(masks, scores, float(mask_nms_iou))
            return masks[keep_indices], boxes[keep_indices], scores[keep_indices]

    def infer_deep_scan(
        self,
        image: Image.Image,
        prompt: str,
        primary_confidence: float,
        candidate_confidence: float,
        min_area_pixels: int,
        max_area_fraction: float,
        fusion_mask_nms_iou: float,
        grid_rows: int = 2,
        grid_cols: int = 2,
        tile_overlap: float = 0.20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Full-image + overlapping tiled inference for small/occluded fruit.

        The primary layer is validated at ~0.45. We infer down to the candidate
        floor (normally ~0.40) so 0.40-0.45 detections can remain hidden as
        optional low-confidence suggestions without changing the >=0.45 primary
        layer. Fusion filtering and Mask-NMS are applied in full-image space.
        """
        with self._lock:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            candidate_confidence = min(float(candidate_confidence), float(primary_confidence))

            all_masks: list[np.ndarray] = []
            all_boxes: list[np.ndarray] = []
            all_scores: list[np.ndarray] = []
            all_sources: list[str] = []
            full_count = 0
            tile_raw_count = 0
            total_seconds = 0.0

            full_masks, full_boxes, full_scores, elapsed = self._infer_text_raw(
                image=image,
                prompt=prompt,
                confidence_threshold=candidate_confidence,
            )
            total_seconds += elapsed
            if len(full_masks):
                all_masks.extend(list(full_masks))
                all_boxes.extend(list(full_boxes))
                all_scores.extend([np.asarray([score], dtype=np.float32) for score in full_scores])
                all_sources.extend(["full"] * len(full_masks))
                full_count = len(full_masks)

            tiles = generate_overlapping_tiles(
                width=width,
                height=height,
                rows=int(grid_rows),
                cols=int(grid_cols),
                overlap_fraction=float(tile_overlap),
            )

            for tile in tiles:
                x0, y0, x1, y1 = tile["x0"], tile["y0"], tile["x1"], tile["y1"]
                crop = image.crop((x0, y0, x1, y1))
                tile_masks, tile_boxes, tile_scores, elapsed = self._infer_text_raw(
                    image=crop,
                    prompt=prompt,
                    confidence_threshold=candidate_confidence,
                )
                total_seconds += elapsed
                tile_raw_count += len(tile_masks)

                for mask, box, score in zip(tile_masks, tile_boxes, tile_scores):
                    full_mask = np.zeros((height, width), dtype=bool)
                    full_mask[y0:y1, x0:x1] = mask
                    mapped_box = np.asarray(
                        [
                            np.clip(float(box[0]) + x0, 0, width),
                            np.clip(float(box[1]) + y0, 0, height),
                            np.clip(float(box[2]) + x0, 0, width),
                            np.clip(float(box[3]) + y0, 0, height),
                        ],
                        dtype=np.float32,
                    )
                    all_masks.append(full_mask)
                    all_boxes.append(mapped_box)
                    all_scores.append(np.asarray([score], dtype=np.float32))
                    all_sources.append("tile")

            if not all_masks:
                info = {
                    "mode": "Deep Scan",
                    "tiles": len(tiles),
                    "raw_full_predictions": 0,
                    "raw_tile_predictions": 0,
                    "retained_predictions": 0,
                    "inference_seconds": total_seconds,
                }
                return (
                    np.zeros((0, height, width), dtype=bool),
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=object),
                    info,
                )

            masks = np.stack(all_masks, axis=0).astype(bool, copy=False)
            boxes = np.asarray(all_boxes, dtype=np.float32).reshape(-1, 4)
            scores = np.asarray([float(v.reshape(-1)[0]) for v in all_scores], dtype=np.float32)
            sources = np.asarray(all_sources, dtype=object)

            areas = masks.reshape(len(masks), -1).sum(axis=1)
            max_area_pixels = float(max_area_fraction) * float(width * height)
            keep = (
                (scores >= candidate_confidence)
                & (areas >= int(min_area_pixels))
                & (areas <= max_area_pixels)
            )
            masks = masks[keep]
            boxes = boxes[keep]
            scores = scores[keep]
            sources = sources[keep]

            nms_keep = deep_mask_nms_indices(
                masks=masks,
                scores=scores,
                sources=sources,
                iou_threshold=float(fusion_mask_nms_iou),
            )
            masks = masks[nms_keep]
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]
            sources = sources[nms_keep]

            # Recompute boxes from the final full-resolution masks. This avoids
            # any crop-boundary rounding discrepancy in the displayed annotation.
            if len(masks):
                boxes = np.asarray([bbox_from_mask(mask) for mask in masks], dtype=np.float32)

            info = {
                "mode": "Deep Scan",
                "tiles": len(tiles),
                "raw_full_predictions": int(full_count),
                "raw_tile_predictions": int(tile_raw_count),
                "retained_predictions": int(len(masks)),
                "primary_predictions": int(np.sum(scores >= float(primary_confidence))),
                "candidate_predictions": int(np.sum(scores < float(primary_confidence))),
                "inference_seconds": float(total_seconds),
                "grid": f"{grid_rows}x{grid_cols}",
                "tile_overlap": float(tile_overlap),
                "fusion_nms_iou": float(fusion_mask_nms_iou),
            }
            return masks, boxes, scores, sources, info

    def interactive_mask(
        self,
        image: Image.Image,
        point_coords: list[tuple[float, float]],
        point_labels: list[int],
        box_xyxy: list[float] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Generate/refine one instance using SAM 3's interactive predictor."""
        import torch

        if not point_coords and box_xyxy is None:
            raise ValueError("At least one positive/negative click or a box prompt is required.")
        if len(point_coords) != len(point_labels):
            raise ValueError("point_coords and point_labels length mismatch.")

        with self._lock:
            self.load()
            predictor = self.model.inst_interactive_predictor
            image = ImageOps.exif_transpose(image).convert("RGB")

            # Box-only SAM 3 prompts must use None for point coordinates/labels.
            # Empty arrays are not a substitute in some official predictor builds.
            coords = np.asarray(point_coords, dtype=np.float32) if point_coords else None
            labels = np.asarray(point_labels, dtype=np.int32) if point_labels else None
            box = None if box_xyxy is None else np.asarray(box_xyxy, dtype=np.float32)

            with torch.inference_mode():
                with self._autocast_context():
                    predictor.set_image(image)
                    masks, quality_scores, _low_res = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        box=box,
                        multimask_output=True,
                    )

            masks = tensor_to_numpy(masks)
            quality_scores = tensor_to_numpy(quality_scores).reshape(-1)

            if masks.ndim == 4 and masks.shape[0] == 1:
                masks = masks[0]
            if masks.ndim == 2:
                masks = masks[None, ...]

            if len(masks) == 0:
                raise RuntimeError("Interactive SAM 3 returned no masks.")

            if len(quality_scores) != len(masks):
                quality_scores = np.zeros((len(masks),), dtype=np.float32)

            best = int(np.argmax(quality_scores)) if len(quality_scores) else 0
            mask = np.asarray(masks[best]) > 0
            score = float(quality_scores[best]) if len(quality_scores) else 0.0

            if int(mask.sum()) == 0:
                raise RuntimeError("Interactive SAM 3 returned an empty mask.")

            return mask, score


# =============================================================================
# Runtime sessions
# =============================================================================


@dataclass
class RuntimeSession:
    manager: AnnotationManager
    image: Image.Image
    image_name: str
    inference_mode: str = "Fast"
    inference_info: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    refine_points: dict[int, list[tuple[float, float, int]]] = field(default_factory=dict)


SESSIONS: dict[str, RuntimeSession] = {}
SESSIONS_LOCK = threading.RLock()


def new_session_id() -> str:
    return uuid.uuid4().hex


def put_session(session_id: str, session: RuntimeSession) -> None:
    with SESSIONS_LOCK:
        session.session_id = session_id
        SESSIONS[session_id] = session


def get_session(session_id: str) -> RuntimeSession:
    if not session_id:
        raise RuntimeError("No active annotation session. Run Auto Annotation first.")
    with SESSIONS_LOCK:
        if session_id not in SESSIONS:
            raise RuntimeError("The annotation session expired. Run Auto Annotation again.")
        return SESSIONS[session_id]


# =============================================================================
# B2: validation and backend geometry transaction (testable without Gradio)
# =============================================================================


def apply_box_edit_transaction(
    session: RuntimeSession,
    runtime: Sam3Runtime,
    *,
    instance_id: int,
    box_xyxy: Any,
    original_box: Any,
    expected_revision: int,
) -> dict[str, Any]:
    """Use a validated client rectangle as a SAM 3 prompt and commit its mask.

    Never treat the submitted rectangle as COCO ground truth. The returned
    instance bbox is recalculated from the new mask inside AnnotationManager.
    No mutation occurs if the predictor or validation fails.
    """
    manager = session.manager
    if expected_revision != len(manager.history):
        raise ValueError("Annotations changed while editing. Reopen this instance and retry.")
    instance = manager.get(int(instance_id))

    if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) != 4:
        raise ValueError("Edited rectangle requires exactly four xyxy values.")
    if not isinstance(original_box, (list, tuple)) or len(original_box) != 4:
        raise ValueError("Original rectangle is missing or invalid.")
    box = [float(v) for v in box_xyxy]
    old_claim = [float(v) for v in original_box]
    if any(not math.isfinite(v) for v in box + old_claim):
        raise ValueError("Rectangle coordinates must be finite.")
    if any(abs(a-b) > 0.55 for a,b in zip(old_claim, instance.bbox_xyxy)):
        raise ValueError("This box changed since editing started. Please retry.")

    W, H = session.image.size
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1-x0 < 4 or y1-y0 < 4:
        raise ValueError("Edited rectangle is outside image bounds or smaller than 4 px.")
    if max(abs(a-b) for a,b in zip(box, old_claim)) < 0.5:
        raise ValueError("Box has not changed. Move or resize it before applying.")

    # Potentially slow GPU inference: no state changed until it succeeds.
    mask, mask_quality = runtime.interactive_mask(
        image=session.image,
        point_coords=[],
        point_labels=[],
        box_xyxy=box,
    )
    if expected_revision != len(manager.history):
        raise ValueError("Annotations changed during SAM 3 inference. Result discarded.")
    # The manager validates the mask and records both user box and mask bbox.
    edited = manager.apply_box_refinement(
        int(instance_id), mask=mask, requested_box_xyxy=box,
        sam_quality=mask_quality,
    )
    session.refine_points.pop(int(instance_id), None)
    return {
        "instance_id": int(edited.instance_id),
        "requested_box_xyxy": box,
        "final_bbox_xyxy": [float(v) for v in edited.bbox_xyxy],
        "new_mask_area": int(edited.area),
        "sam_mask_quality": float(mask_quality),
    }


# =============================================================================
# Visualization
# =============================================================================


def stable_color(label: str, instance_id: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(f"{label}:{instance_id}".encode("utf-8")).digest()
    return (
        60 + digest[0] % 176,
        60 + digest[1] % 176,
        60 + digest[2] % 176,
    )


def compute_mask_boundary(mask: np.ndarray) -> np.ndarray:
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
    return center & (~interior)


def draw_text_box(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: tuple[int, int, int]) -> None:
    x = max(0, int(x))
    y = max(0, int(y))
    try:
        box = draw.textbbox((x, y), text)
        padding = 2
        draw.rectangle(
            (
                box[0] - padding,
                box[1] - padding,
                box[2] + padding,
                box[3] + padding,
            ),
            fill=(0, 0, 0),
        )
    except Exception:
        pass
    draw.text((x, y), text, fill=fill)


def render_overlay(
    image: Image.Image,
    manager: AnnotationManager,
    show_candidates: bool,
    selected_id: int | None,
    *,
    draw_static_boxes_and_labels: bool = True,
) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    base = np.asarray(image, dtype=np.float32).copy()

    annotations = manager.active_annotations(show_candidates=show_candidates)

    for ann in annotations:
        color = np.asarray(stable_color(ann.class_label, ann.instance_id), dtype=np.float32)

        if ann.tier == "candidate":
            alpha = 0.24
        elif ann.review_status in {"accepted", "corrected"}:
            alpha = 0.55
        else:
            alpha = 0.45

        base[ann.mask] = (1.0 - alpha) * base[ann.mask] + alpha * color
        boundary = compute_mask_boundary(ann.mask)
        base[boundary] = color

        if selected_id is not None and ann.instance_id == selected_id:
            # Thicken the selected contour by dilating the boundary one pixel.
            selected = boundary.copy()
            selected[:-1] |= boundary[1:]
            selected[1:] |= boundary[:-1]
            selected[:, :-1] |= boundary[:, 1:]
            selected[:, 1:] |= boundary[:, :-1]
            base[selected] = np.asarray((255, 255, 255), dtype=np.float32)

    result = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    # B1 uses genuine DOM annotation frames + clickable label buttons.
    # Static PIL labels are retained as an optional fallback but are disabled
    # for the interactive canvas to prevent duplicate/non-clickable text.
    if not draw_static_boxes_and_labels:
        return result

    draw = ImageDraw.Draw(result)

    for ann in annotations:
        x0, y0, x1, y1 = ann.bbox_xyxy
        color = stable_color(ann.class_label, ann.instance_id)
        width = 4 if selected_id == ann.instance_id else 2
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)

        score_text = "manual" if ann.score is None else f"{ann.score:.2f}"
        tier_text = "LOW" if ann.tier == "candidate" else ann.review_status.upper()
        label = f"{ann.class_label} #{ann.instance_id} {score_text} [{tier_text}]"
        draw_text_box(draw, int(x0 + 2), int(max(0, y0 - 14)), label, color)

    return result


def format_status(manager: AnnotationManager, message: str = "") -> str:
    stats = manager.stats()
    prefix = f"**{message}**\n\n" if message else ""
    return (
        prefix
        + f"Active: **{stats['active']}** | Primary: **{stats['primary']}** | "
        + f"Low-confidence: **{stats['candidate']}** | Manual: **{stats['manual']}** | "
        + f"Accepted: **{stats['accepted']}** | Corrected: **{stats['corrected']}** | "
        + f"Deleted: **{stats['deleted']}**"
    )


def safe_selected_id(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        integer = int(float(value))
        return integer if integer > 0 else None
    except Exception:
        return None


def ensure_image_name(name: str | None) -> str:
    text = Path(str(name or "annotation_image.png")).name
    if not Path(text).suffix:
        text += ".png"
    return text


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    if intersection == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union else 0.0


# =============================================================================
# Gradio application
# =============================================================================


def build_app(
    project_root: Path,
    runtime: Sam3Runtime,
    defaults: dict[str, Any],
    deep_defaults: dict[str, Any],
):
    import gradio as gr

    # B1 depends on the native custom HTML event API introduced in Gradio 6.
    if int(str(gr.__version__).split(".")[0]) < 6:
        raise RuntimeError(
            "Inline canvas requires Gradio >= 6. "
            "Install using: python -m pip install 'gradio>=6,<7'"
        )

    table_headers = ["ID", "Class", "Score", "Tier", "Review status", "Area px", "Source", "BBox xyxy"]

    def add_predictions_to_manager(
        manager: AnnotationManager,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_label: str,
        sources: np.ndarray | list[str] | None = None,
    ) -> int:
        if sources is None:
            sources = ["sam3_text_fast"] * len(masks)
        created = 0
        for index, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            score_value = float(score)
            if score_value < manager.candidate_threshold:
                continue
            tier = "primary" if score_value >= manager.primary_threshold else "candidate"
            raw_source = str(sources[index]) if index < len(sources) else "sam3_text"
            if raw_source == "full":
                source = "sam3_deep_full"
            elif raw_source == "tile":
                source = "sam3_deep_tile"
            else:
                source = raw_source
            manager.add_instance(
                mask=mask,
                class_label=class_label,
                score=score_value,
                bbox_xyxy=box,
                source=source,
                tier=tier,
                review_status="pending",
            )
            created += 1
        return created

    def run_mode_inference(
        image: Image.Image,
        prompt: str,
        primary: float,
        candidate: float,
        mode: str,
    ):
        if str(mode) == "Deep Scan":
            masks, boxes, scores, sources, info = runtime.infer_deep_scan(
                image=image,
                prompt=prompt,
                primary_confidence=primary,
                candidate_confidence=candidate,
                min_area_pixels=int(deep_defaults["min_area_pixels"]),
                max_area_fraction=float(deep_defaults["max_area_fraction"]),
                fusion_mask_nms_iou=float(deep_defaults["fusion_mask_nms_iou"]),
                grid_rows=int(deep_defaults["grid_rows"]),
                grid_cols=int(deep_defaults["grid_cols"]),
                tile_overlap=float(deep_defaults["tile_overlap"]),
            )
            return masks, boxes, scores, sources, info

        start = time.perf_counter()
        masks, boxes, scores = runtime.infer_text(
            image=image,
            prompt=prompt,
            candidate_confidence=candidate,
            min_area_pixels=int(defaults["min_area_pixels"]),
            max_area_fraction=float(defaults["max_area_fraction"]),
            mask_nms_iou=float(defaults["mask_nms_iou"]),
        )
        elapsed = time.perf_counter() - start
        sources = np.asarray(["sam3_text_fast"] * len(masks), dtype=object)
        info = {
            "mode": "Fast",
            "retained_predictions": int(len(masks)),
            "primary_predictions": int(np.sum(scores >= float(primary))) if len(scores) else 0,
            "candidate_predictions": int(np.sum(scores < float(primary))) if len(scores) else 0,
            "inference_seconds": float(elapsed),
        }
        return masks, boxes, scores, sources, info

    def output_bundle(
        session: RuntimeSession,
        show_candidates: bool,
        selected_id: int | None,
        message: str,
    ):
        overlay = render_overlay(
            session.image,
            session.manager,
            bool(show_candidates),
            selected_id,
            draw_static_boxes_and_labels=False,
        )
        canvas_payload = make_canvas_payload(
            overlay,
            session.manager,
            session_id=session.session_id,
            selected_id=selected_id,
            show_candidates=bool(show_candidates),
        )
        rows = session.manager.table_rows(show_candidates=bool(show_candidates))
        for row in rows:
            box = session.manager.get(int(row[0])).bbox_xyxy
            row.append(", ".join(str(int(round(v))) for v in box))
        return canvas_payload, rows, format_status(session.manager, message), selected_id or 0

    def cb_mode_change(mode: str):
        if str(mode) == "Deep Scan":
            return (
                float(deep_defaults["primary_confidence"]),
                float(deep_defaults["candidate_confidence"]),
                (
                    "**Deep Scan:** full image + "
                    f"{deep_defaults['grid_rows']}x{deep_defaults['grid_cols']} overlapping tiles "
                    f"({deep_defaults['tile_overlap']:.0%} overlap). Designed to recover small "
                    "and partially visible fruit. It is slower but substantially improves recall."
                ),
            )
        return (
            float(defaults["primary_confidence"]),
            float(defaults["candidate_confidence"]),
            "**Fast mode:** full-image inference; recommended for normal images and rapid review.",
        )

    def cb_load_model(candidate_confidence: float):
        try:
            return runtime.load(float(candidate_confidence))
        except Exception as error:
            return f"MODEL LOAD FAILED: {error}"

    def cb_run_auto(
        image,
        image_name,
        project_name,
        class_label,
        prompt,
        inference_mode,
        primary_confidence,
        candidate_confidence,
        show_candidates,
    ):
        try:
            if image is None:
                raise ValueError("Upload an image first.")

            image = ImageOps.exif_transpose(image).convert("RGB")
            image_name = ensure_image_name(image_name)
            class_label = str(class_label).strip()
            prompt = str(prompt).strip()

            if not class_label:
                raise ValueError("Class label cannot be empty.")
            if not prompt:
                raise ValueError("SAM 3 text prompt cannot be empty.")

            primary = float(primary_confidence)
            candidate = float(candidate_confidence)
            if not 0.0 <= candidate <= primary <= 1.0:
                raise ValueError(
                    "Require 0 <= candidate confidence <= primary confidence <= 1."
                )

            masks, boxes, scores, sources, inference_info = run_mode_inference(
                image=image,
                prompt=prompt,
                primary=primary,
                candidate=candidate,
                mode=str(inference_mode),
            )

            manager = AnnotationManager(
                project_root=project_root,
                project_name=str(project_name or "apple_demo"),
                primary_threshold=primary,
                candidate_threshold=candidate,
            )
            manager.set_image(image_name=image_name, width=image.width, height=image.height)
            manager.save_source_image(image)
            created = add_predictions_to_manager(
                manager=manager,
                masks=masks,
                boxes=boxes,
                scores=scores,
                class_label=class_label,
                sources=sources,
            )
            auto_json = manager.save_snapshot("auto", image=image)

            session_id = new_session_id()
            session = RuntimeSession(
                manager=manager,
                image=image,
                image_name=image_name,
                inference_mode=str(inference_mode),
                inference_info=dict(inference_info),
            )
            put_session(session_id, session)

            overlay, rows, status, selected = output_bundle(
                session,
                bool(show_candidates),
                None,
                f"{inference_mode} auto annotation complete: {created} retained instances "
                f"({inference_info.get('primary_predictions', 0)} primary, "
                f"{inference_info.get('candidate_predictions', 0)} low-confidence) in "
                f"{inference_info.get('inference_seconds', 0.0):.2f}s. "
                f"Auto snapshot saved to {auto_json.name}.",
            )

            return overlay, rows, status, selected, session_id, runtime.status_text()

        except Exception as error:
            return None, [], f"**AUTO ANNOTATION FAILED:** {error}", 0, "", runtime.status_text()

    def cb_add_class_pass(
        session_id,
        class_label,
        prompt,
        inference_mode,
        primary_confidence,
        candidate_confidence,
        show_candidates,
    ):
        try:
            session = get_session(session_id)
            label = str(class_label).strip()
            prompt = str(prompt).strip()
            if not label or not prompt:
                raise ValueError("Class label and prompt are required.")

            primary = float(primary_confidence)
            candidate = float(candidate_confidence)
            if not 0.0 <= candidate <= primary <= 1.0:
                raise ValueError("Invalid confidence thresholds.")

            session.manager.primary_threshold = primary
            session.manager.candidate_threshold = candidate

            masks, boxes, scores, sources, inference_info = run_mode_inference(
                image=session.image,
                prompt=prompt,
                primary=primary,
                candidate=candidate,
                mode=str(inference_mode),
            )

            existing_same_class = [
                ann
                for ann in session.manager.active_annotations(show_candidates=True)
                if ann.class_label == label
            ]

            added = 0
            skipped_duplicates = 0
            for index, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                if any(mask_iou(mask, ann.mask) >= 0.70 for ann in existing_same_class):
                    skipped_duplicates += 1
                    continue

                tier = "primary" if float(score) >= primary else "candidate"
                raw_source = str(sources[index]) if index < len(sources) else "sam3_text"
                if raw_source == "full":
                    source = "sam3_deep_full"
                elif raw_source == "tile":
                    source = "sam3_deep_tile"
                else:
                    source = raw_source
                new_ann = session.manager.add_instance(
                    mask=mask,
                    class_label=label,
                    score=float(score),
                    bbox_xyxy=box,
                    source=source,
                    tier=tier,
                    review_status="pending",
                )
                existing_same_class.append(new_ann)
                added += 1

            session.manager.save_snapshot("auto", image=session.image)
            return output_bundle(
                session,
                bool(show_candidates),
                None,
                f"Added {inference_mode} class pass '{label}': {added} new instances; "
                f"{skipped_duplicates} same-class duplicates skipped. "
                f"Inference {inference_info.get('inference_seconds', 0.0):.2f}s.",
            )
        except Exception as error:
            try:
                session = get_session(session_id)
                return output_bundle(session, bool(show_candidates), None, f"Add-class failed: {error}")
            except Exception:
                return None, [], f"**ADD CLASS FAILED:** {error}", 0

    def cb_image_click(
        session_id,
        selected_id_value,
        tool_mode,
        class_label,
        show_candidates,
        evt: gr.SelectData,
    ):
        try:
            session = get_session(session_id)
            selected_id = safe_selected_id(selected_id_value)

            index = evt.index
            if not isinstance(index, (list, tuple)) or len(index) < 2:
                raise ValueError(f"Could not read click coordinates from Gradio event: {index}")

            x = float(index[0])
            y = float(index[1])
            mode = str(tool_mode)

            if mode == "Select instance":
                hit = session.manager.find_instance_at(
                    int(round(x)),
                    int(round(y)),
                    show_candidates=bool(show_candidates),
                )
                if hit is None:
                    return output_bundle(
                        session,
                        bool(show_candidates),
                        None,
                        f"No visible annotation at ({int(x)}, {int(y)}).",
                    )
                return output_bundle(
                    session,
                    bool(show_candidates),
                    hit.instance_id,
                    f"Selected {hit.class_label} #{hit.instance_id}.",
                )

            if mode == "Add missing (+)":
                label = str(class_label).strip() or "object"
                mask, score = runtime.interactive_mask(
                    image=session.image,
                    point_coords=[(x, y)],
                    point_labels=[1],
                    box_xyxy=None,
                )
                ann = session.manager.add_instance(
                    mask=mask,
                    class_label=label,
                    score=score,
                    bbox_xyxy=bbox_from_mask(mask),
                    source="sam3_click_add",
                    tier="manual",
                    review_status="corrected",
                )
                return output_bundle(
                    session,
                    bool(show_candidates),
                    ann.instance_id,
                    f"Added missing {label} #{ann.instance_id} from positive click.",
                )

            if mode in {"Refine (+)", "Refine (-)"}:
                if selected_id is None:
                    raise ValueError("Select an instance before refining it.")

                ann = session.manager.get(selected_id)
                label_value = 1 if mode == "Refine (+)" else 0
                points = session.refine_points.setdefault(selected_id, [])
                points.append((x, y, label_value))

                coords = [(px, py) for px, py, _label in points]
                labels = [label for _px, _py, label in points]

                mask, score = runtime.interactive_mask(
                    image=session.image,
                    point_coords=coords,
                    point_labels=labels,
                    box_xyxy=ann.bbox_xyxy,
                )
                session.manager.replace_mask(
                    selected_id,
                    mask=mask,
                    score=score,
                    source="sam3_click_refine",
                )

                positives = sum(label == 1 for label in labels)
                negatives = sum(label == 0 for label in labels)
                return output_bundle(
                    session,
                    bool(show_candidates),
                    selected_id,
                    f"Refined #{selected_id} using {positives} positive and "
                    f"{negatives} negative click(s).",
                )

            raise ValueError(f"Unknown click tool: {mode}")

        except Exception as error:
            try:
                session = get_session(session_id)
                return output_bundle(
                    session,
                    bool(show_candidates),
                    safe_selected_id(selected_id_value),
                    f"Click action failed: {error}",
                )
            except Exception:
                return None, [], f"**CLICK ACTION FAILED:** {error}", 0

    def cb_accept(session_id, selected_id_value, show_candidates):
        try:
            session = get_session(session_id)
            selected_id = safe_selected_id(selected_id_value)
            if selected_id is None:
                raise ValueError("Select an instance first.")
            session.manager.accept(selected_id)
            return output_bundle(
                session,
                bool(show_candidates),
                selected_id,
                f"Accepted instance #{selected_id}.",
            )
        except Exception as error:
            session = get_session(session_id)
            return output_bundle(session, bool(show_candidates), None, f"Accept failed: {error}")

    def cb_accept_all_primary(session_id, selected_id_value, show_candidates):
        try:
            session = get_session(session_id)
            count = 0
            for ann in list(session.manager.active_annotations(show_candidates=True)):
                if ann.tier == "primary" and ann.review_status == "pending":
                    session.manager.accept(ann.instance_id)
                    count += 1
            return output_bundle(
                session,
                bool(show_candidates),
                safe_selected_id(selected_id_value),
                f"Accepted {count} pending primary annotations.",
            )
        except Exception as error:
            session = get_session(session_id)
            return output_bundle(session, bool(show_candidates), None, f"Accept-all failed: {error}")

    def cb_delete(session_id, selected_id_value, show_candidates):
        try:
            session = get_session(session_id)
            selected_id = safe_selected_id(selected_id_value)
            if selected_id is None:
                raise ValueError("Select an instance first.")
            session.manager.delete(selected_id)
            session.refine_points.pop(selected_id, None)
            return output_bundle(
                session,
                bool(show_candidates),
                None,
                f"Deleted instance #{selected_id}.",
            )
        except Exception as error:
            session = get_session(session_id)
            return output_bundle(session, bool(show_candidates), None, f"Delete failed: {error}")

    def cb_relabel(session_id, selected_id_value, new_label, show_candidates):
        try:
            session = get_session(session_id)
            selected_id = safe_selected_id(selected_id_value)
            if selected_id is None:
                raise ValueError("Select an instance first.")
            session.manager.relabel(selected_id, str(new_label))
            return output_bundle(
                session,
                bool(show_candidates),
                selected_id,
                f"Relabelled instance #{selected_id} as '{str(new_label).strip()}'.",
            )
        except Exception as error:
            session = get_session(session_id)
            return output_bundle(
                session,
                bool(show_candidates),
                safe_selected_id(selected_id_value),
                f"Relabel failed: {error}",
            )

    def cb_reset_refine_points(session_id, selected_id_value, show_candidates):
        try:
            session = get_session(session_id)
            selected_id = safe_selected_id(selected_id_value)
            if selected_id is None:
                raise ValueError("Select an instance first.")
            session.refine_points.pop(selected_id, None)
            return output_bundle(
                session,
                bool(show_candidates),
                selected_id,
                f"Cleared stored refinement clicks for instance #{selected_id}.",
            )
        except Exception as error:
            session = get_session(session_id)
            return output_bundle(
                session,
                bool(show_candidates),
                safe_selected_id(selected_id_value),
                f"Reset failed: {error}",
            )

    def cb_toggle_candidates(session_id, selected_id_value, show_candidates):
        try:
            session = get_session(session_id)
            return output_bundle(
                session,
                bool(show_candidates),
                safe_selected_id(selected_id_value),
                "Low-confidence candidate visibility updated.",
            )
        except Exception as error:
            return None, [], f"**DISPLAY UPDATE FAILED:** {error}", 0

    def cb_save_review(session_id, show_candidates, selected_id_value):
        try:
            session = get_session(session_id)
            reviewed_json = session.manager.save_snapshot("reviewed", image=session.image)
            coco_json = session.manager.export_coco()
            overlay, rows, status, selected = output_bundle(
                session,
                bool(show_candidates),
                safe_selected_id(selected_id_value),
                "Reviewed annotation saved and COCO export generated.",
            )
            return overlay, rows, status, selected, str(reviewed_json), str(coco_json)
        except Exception as error:
            session = get_session(session_id)
            overlay, rows, status, selected = output_bundle(
                session,
                bool(show_candidates),
                safe_selected_id(selected_id_value),
                f"Save/export failed: {error}",
            )
            return overlay, rows, status, selected, None, None


    # ------------------------------------------------------------------
    # B1: label-chip selection and in-place class relabeling
    # ------------------------------------------------------------------

    def cb_label_chip_select(session_id: str, evt: gr.EventData):
        """Select an instance when its real DOM label is clicked.

        Do not redraw the canvas here: redrawing would close the open popover.
        """
        try:
            session = get_session(session_id)
            instance_id = int(evt.instance_id)
            ann = session.manager.get(instance_id)
            return (
                instance_id,
                ann.class_label,
                format_status(session.manager, f"Editing {ann.class_label} #{instance_id} on image."),
            )
        except Exception as error:
            return 0, "", f"**Label selection failed:** {error}"

    def cb_inline_label_save(canvas_value: Any, session_id: str, show_candidates: bool):
        """Validate a canvas edit and apply it ONLY to AnnotationManager.

        Client-submitted masks/boxes/labels outside the requested class are
        ignored. Each relabel is immediately persisted and COCO is regenerated
        so the user never downloads an older label mapping by accident.
        """
        try:
            session = get_session(session_id)
            payload = canvas_value if isinstance(canvas_value, dict) else {}
            action = payload.get("action") or {}
            if action.get("type") != "relabel":
                raise ValueError("No inline relabel action was supplied.")
            if payload.get("session_id") != session.session_id:
                raise ValueError("Stale image session. Please reopen the latest image.")
            if int(payload.get("revision", -1)) != len(session.manager.history):
                raise ValueError("Annotations changed while editing. Please retry on the refreshed canvas.")

            instance_id = int(action.get("instance_id", 0))
            new_label = str(action.get("new_label", "")).strip()
            if not new_label or len(new_label) > 80 or any(ord(ch) < 32 for ch in new_label):
                raise ValueError("Class label must contain 1–80 printable characters.")
            ann = session.manager.get(instance_id)
            old_label = ann.class_label

            if new_label != old_label:
                session.manager.relabel(instance_id, new_label)
                description = f"Relabeled #{instance_id}: '{old_label}' → '{new_label}'."
            else:
                description = f"Instance #{instance_id} already has label '{new_label}'."

            reviewed = session.manager.save_snapshot("reviewed", image=session.image)
            coco = session.manager.export_coco()
            canvas, rows, status, selected = output_bundle(
                session, bool(show_candidates), instance_id,
                description + " Reviewed snapshot and COCO export updated.",
            )
            return canvas, rows, status, selected, new_label, str(reviewed), str(coco)
        except Exception as error:
            try:
                session = get_session(session_id)
                canvas, rows, status, selected = output_bundle(
                    session, bool(show_candidates), None, f"Inline label edit rejected: {error}",
                )
                return canvas, rows, status, selected, "", None, None
            except Exception:
                return empty_canvas_payload(), [], f"**Inline label edit failed:** {error}", 0, "", None, None

    # ------------------------------------------------------------------
    # B2: drag / resize a visible box and apply SAM 3 box-guided refinement
    # ------------------------------------------------------------------

    def cb_inline_box_save(canvas_value: Any, session_id: str, show_candidates: bool):
        """Validate canvas draft, rerun SAM 3 with a box prompt, persist mask+COCO.

        A drawing-only edit never becomes a stored bbox: SAM 3 generates a new
        mask; the authoritative bbox is the tight external box of that mask.
        """
        committed = False
        try:
            session = get_session(session_id)
            payload = canvas_value if isinstance(canvas_value, dict) else {}
            action = payload.get("action") or {}
            if action.get("type") != "bbox_refine":
                raise ValueError("No bounding-box edit action supplied.")
            if payload.get("session_id") != session.session_id:
                raise ValueError("Stale image session. Reopen the newest image.")
            expected_revision = int(payload.get("revision", -1))
            instance_id = int(action.get("instance_id", 0))
            info = apply_box_edit_transaction(
                session, runtime,
                instance_id=instance_id,
                box_xyxy=action.get("box_xyxy"),
                original_box=action.get("original_box"),
                expected_revision=expected_revision,
            )
            committed = True
            # Save both artifacts before telling the user the edit is complete.
            reviewed = session.manager.save_snapshot("reviewed", image=session.image)
            coco = session.manager.export_coco()
            canvas, rows, status, selected = output_bundle(
                session, bool(show_candidates), instance_id,
                f"Box-guided SAM 3 refinement applied to #{instance_id}. "
                f"New mask area: {info['new_mask_area']} px. "
                f"Actual bbox follows the new mask. Reviewed/COCO updated.",
            )
            ann = session.manager.get(instance_id)
            return canvas, rows, status, selected, ann.class_label, str(reviewed), str(coco)
        except Exception as error:
            try:
                session = get_session(session_id)
                canvas, rows, status, selected = output_bundle(
                    session, bool(show_candidates), None,
                    (
                        f"Box mask was changed in memory, but saving/export failed: {error}. "
                        "Please use Save Reviewed Annotation + Export COCO to retry."
                        if committed else
                        f"Box edit rejected; previous annotation preserved: {error}"
                    ),
                )
                return canvas, rows, status, selected, "", None, None
            except Exception:
                return empty_canvas_payload(), [], f"**Box edit failed:** {error}", 0, "", None, None

    def cb_canvas_submit(canvas_value: Any, session_id: str, show_candidates: bool):
        payload = canvas_value if isinstance(canvas_value, dict) else {}
        action = payload.get("action") or {}
        if action.get("type") == "bbox_refine":
            return cb_inline_box_save(canvas_value, session_id, show_candidates)
        return cb_inline_label_save(canvas_value, session_id, show_candidates)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    with gr.Blocks(title="AgriAnnotationSystem") as demo:
        gr.Markdown(
            "# AgriAnnotationSystem\n"
            "SAM 3 assisted agricultural instance annotation with human review. "
            "Choose **Fast** for normal images or **Deep Scan** for dense, small, "
            "partially visible or leaf-occluded fruit."
        )

        gr.Markdown(
            f"**Fast (Stage 4B):** prompt=`{defaults['prompt']}`, "
            f"primary={defaults['primary_confidence']:.2f}, "
            f"candidate={defaults['candidate_confidence']:.2f}, "
            f"mask_NMS={defaults['mask_nms_iou']:.2f}.  "
            f"Source: `{defaults['source']}`\n\n"
            f"**Deep Scan (Stage 4C single-prompt tiled):** prompt=`{deep_defaults['prompt']}`, "
            f"primary={deep_defaults['primary_confidence']:.2f}, "
            f"candidate={deep_defaults['candidate_confidence']:.2f}, "
            f"grid={deep_defaults['grid_rows']}x{deep_defaults['grid_cols']}, "
            f"overlap={deep_defaults['tile_overlap']:.0%}, "
            f"fusion_NMS={deep_defaults['fusion_mask_nms_iou']:.2f}.  "
            f"Source: `{deep_defaults['source']}`"
        )

        session_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Input agricultural image",
                )
                image_name = gr.Textbox(
                    value="orchard_image.png",
                    label="Image filename",
                )
                project_name = gr.Textbox(
                    value="apple_demo",
                    label="Annotation project",
                )

                with gr.Row():
                    class_label = gr.Textbox(
                        value=defaults["class_label"],
                        label="Output class label",
                    )
                    text_prompt = gr.Textbox(
                        value=defaults["prompt"],
                        label="SAM 3 text prompt",
                    )

                inference_mode = gr.Radio(
                    choices=["Fast", "Deep Scan"],
                    value="Fast",
                    label="Detection mode",
                    info=(
                        "Fast: full-image inference. Deep Scan: full image + 2x2 "
                        "overlapping tiles for better small/occluded-fruit recall."
                    ),
                )

                mode_note = gr.Markdown(
                    "**Fast mode:** recommended for normal images; lowest latency."
                )

                with gr.Accordion("Inference settings", open=False):
                    primary_confidence = gr.Slider(
                        0.0,
                        1.0,
                        value=float(defaults["primary_confidence"]),
                        step=0.01,
                        label="Primary confidence",
                    )
                    candidate_confidence = gr.Slider(
                        0.0,
                        1.0,
                        value=float(defaults["candidate_confidence"]),
                        step=0.01,
                        label="Low-confidence candidate threshold",
                    )
                    show_candidates = gr.Checkbox(
                        value=False,
                        label="Show low-confidence candidates",
                    )

                model_status = gr.Textbox(
                    value="SAM 3 model: not loaded",
                    label="Model status",
                    interactive=False,
                )

                with gr.Row():
                    load_model_button = gr.Button("Load SAM 3", variant="secondary")
                    run_button = gr.Button("Run / Replace Auto Annotation", variant="primary")

                add_class_button = gr.Button(
                    "Add Another Class Pass",
                    variant="secondary",
                )

            with gr.Column(scale=2):
                gr.Markdown(
                    "### B2 — Interactive Annotation Canvas\n"
                    "**Click a label** to relabel or edit its box. "
                    "To adjust geometry, select an instance, choose **Edit selected box**, "
                    "then drag the rectangle or its eight resize handles. "
                    "Click **Apply box with SAM 3** to regenerate the mask, or **Cancel** "
                    "to keep the original. Other image-click tools remain available."
                )
                annotated_view = gr.HTML(
                    value=empty_canvas_payload(),
                    html_template=CANVAS_HTML,
                    css_template=CANVAS_CSS,
                    js_on_load=CANVAS_JS,
                    show_label=False,
                    container=False,
                    elem_id="agri-b2-interactive-canvas",
                )

                tool_mode = gr.Radio(
                    choices=[
                        "Select instance",
                        "Add missing (+)",
                        "Refine (+)",
                        "Refine (-)",
                    ],
                    value="Select instance",
                    label="Image click tool",
                )

                selected_id = gr.Number(
                    value=0,
                    precision=0,
                    label="Selected instance ID",
                )

                with gr.Row():
                    accept_button = gr.Button("Accept Selected")
                    accept_all_button = gr.Button("Accept All Primary")
                    delete_button = gr.Button("Delete Selected", variant="stop")

                with gr.Row():
                    relabel_text = gr.Textbox(value="apple", label="New label")
                    relabel_button = gr.Button("Relabel Selected")
                    reset_refine_button = gr.Button("Reset Refine Clicks")

                annotation_table = gr.Dataframe(
                    headers=table_headers,
                    datatype=["number", "str", "str", "str", "str", "number", "str", "str"],
                    value=[],
                    interactive=False,
                    label="Visible annotation instances",
                    wrap=True,
                )

                status_md = gr.Markdown("No active annotation session.")

                save_review_button = gr.Button(
                    "Save Reviewed Annotation + Export COCO",
                    variant="primary",
                )

                with gr.Row():
                    reviewed_file = gr.File(label="Reviewed snapshot JSON")
                    coco_file = gr.File(label="COCO export JSON")

        # ------------------------------------------------------------------
        # Events
        # ------------------------------------------------------------------

        inference_mode.change(
            cb_mode_change,
            inputs=[inference_mode],
            outputs=[primary_confidence, candidate_confidence, mode_note],
        )

        load_model_button.click(
            cb_load_model,
            inputs=[candidate_confidence],
            outputs=[model_status],
        )

        run_button.click(
            cb_run_auto,
            inputs=[
                input_image,
                image_name,
                project_name,
                class_label,
                text_prompt,
                inference_mode,
                primary_confidence,
                candidate_confidence,
                show_candidates,
            ],
            outputs=[
                annotated_view,
                annotation_table,
                status_md,
                selected_id,
                session_state,
                model_status,
            ],
        )

        add_class_button.click(
            cb_add_class_pass,
            inputs=[
                session_state,
                class_label,
                text_prompt,
                inference_mode,
                primary_confidence,
                candidate_confidence,
                show_candidates,
            ],
            outputs=[annotated_view, annotation_table, status_md, selected_id],
        )

        # The HTML canvas sends real image coordinates, preserving the original
        # point-driven Select/Add/Refine behavior without relying on gr.Image.
        annotated_view.select(
            cb_image_click,
            inputs=[
                session_state,
                selected_id,
                tool_mode,
                class_label,
                show_candidates,
            ],
            outputs=[annotated_view, annotation_table, status_md, selected_id],
        )

        # Clicking a real label opens its editor in JavaScript and only sends
        # the selected instance ID to the existing Gradio controls.
        annotated_view.edit(
            cb_label_chip_select,
            inputs=[session_state],
            outputs=[selected_id, relabel_text, status_md],
        )

        # Submitting the popover validates and saves the class change on the
        # Python backend, then refreshes both image and table from one state.
        annotated_view.submit(
            cb_canvas_submit,
            inputs=[annotated_view, session_state, show_candidates],
            outputs=[
                annotated_view, annotation_table, status_md, selected_id,
                relabel_text, reviewed_file, coco_file,
            ],
        )

        accept_button.click(
            cb_accept,
            inputs=[session_state, selected_id, show_candidates],
            outputs=[annotated_view, annotation_table, status_md, selected_id],
        )

        accept_all_button.click(
            cb_accept_all_primary,
            inputs=[session_state, selected_id, show_candidates],
            outputs=[annotated_view, annotation_table, status_md, selected_id],
        )

        delete_button.click(
            cb_delete,
            inputs=[session_state, selected_id, show_candidates],
            outputs=[annotated_view, annotation_table, status_md, selected_id],
        )

        relabel_button.click(
            cb_relabel,
            inputs=[session_state, selected_id, relabel_text, show_candidates],
            outputs=[annotated_view, annotation_table, status_md, selected_id],
        )

        reset_refine_button.click(
            cb_reset_refine_points,
            inputs=[session_state, selected_id, show_candidates],
            outputs=[annotated_view, annotation_table, status_md, selected_id],
        )

        show_candidates.change(
            cb_toggle_candidates,
            inputs=[session_state, selected_id, show_candidates],
            outputs=[annotated_view, annotation_table, status_md, selected_id],
        )

        save_review_button.click(
            cb_save_review,
            inputs=[session_state, show_candidates, selected_id],
            outputs=[
                annotated_view,
                annotation_table,
                status_md,
                selected_id,
                reviewed_file,
                coco_file,
            ],
        )

    return demo


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the AgriAnnotationSystem SAM 3 human-review UI."
    )
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--no-hf-download",
        action="store_true",
        help="Require a local SAM 3 checkpoint; do not use Hugging Face auto-download.",
    )
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()

    project_root = resolve_project_root(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    defaults = load_optimized_defaults(project_root)
    deep_defaults = load_deep_scan_defaults(project_root, defaults)
    runtime = Sam3Runtime(
        project_root=project_root,
        checkpoint=args.checkpoint,
        allow_hf_download=not args.no_hf_download,
        device=args.device,
    )

    app = build_app(project_root, runtime, defaults, deep_defaults)
    app.queue()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=bool(args.share),
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
