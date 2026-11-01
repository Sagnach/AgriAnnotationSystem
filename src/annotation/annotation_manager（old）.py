from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import json
import shutil

import numpy as np
from PIL import Image


# =============================================================================
# Annotation state manager for AgriAnnotationSystem
# =============================================================================
#
# This module intentionally contains NO SAM 3 model code and NO UI code.
# It owns the annotation state, review actions, persistence, and COCO export.
#
# Internal lifecycle for one instance:
#
#   auto prediction -> pending -> accepted
#                            \-> corrected (relabel/refine)
#                            \-> deleted
#
# Manually added objects start as corrected/manual because a human explicitly
# requested them.
# =============================================================================


VALID_TIERS = {"primary", "candidate", "manual"}
VALID_STATUSES = {"pending", "accepted", "corrected", "deleted"}


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_bool_mask(mask: np.ndarray, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(mask)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(f"Mask must be 2-D; found shape={array.shape}")

    array = array.astype(bool, copy=False)

    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"Mask shape {array.shape} does not match current image shape {expected_shape}."
        )

    return array


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    mask = ensure_bool_mask(mask)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    # XYXY, with x1/y1 treated as exclusive upper bounds.
    return [
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    ]


def bbox_xyxy_to_xywh(box: Iterable[float]) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def normalize_bbox(box: Iterable[float] | None, mask: np.ndarray) -> list[float]:
    if box is None:
        return bbox_from_mask(mask)

    values = [float(v) for v in box]
    if len(values) != 4:
        return bbox_from_mask(mask)

    if values[2] <= values[0] or values[3] <= values[1]:
        return bbox_from_mask(mask)

    return values


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    """Return JSON-safe compressed COCO RLE."""
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")

    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts,
    }


def decode_binary_mask(rle: dict[str, Any]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")

    decoded = mask_utils.decode(
        {
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts,
        }
    )
    return np.asarray(decoded).astype(bool, copy=False)


@dataclass
class AnnotationInstance:
    instance_id: int
    class_label: str
    mask: np.ndarray = field(repr=False)
    score: float | None = None
    bbox_xyxy: list[float] | None = None
    source: str = "sam3_text"
    tier: str = "primary"
    review_status: str = "pending"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    notes: str = ""

    def __post_init__(self) -> None:
        self.instance_id = int(self.instance_id)
        self.class_label = str(self.class_label).strip() or "object"
        self.mask = ensure_bool_mask(self.mask)
        self.score = None if self.score is None else float(self.score)
        self.bbox_xyxy = normalize_bbox(self.bbox_xyxy, self.mask)

        if self.tier not in VALID_TIERS:
            raise ValueError(f"Unsupported annotation tier: {self.tier}")
        if self.review_status not in VALID_STATUSES:
            raise ValueError(f"Unsupported review status: {self.review_status}")

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def is_deleted(self) -> bool:
        return self.review_status == "deleted"

    def to_record(self, include_rle: bool = False, mask_path: str | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "instance_id": self.instance_id,
            "class_label": self.class_label,
            "score": self.score,
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "area_pixels": self.area,
            "source": self.source,
            "tier": self.tier,
            "review_status": self.review_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

        if mask_path is not None:
            record["mask_path"] = mask_path
        if include_rle:
            record["segmentation_rle"] = encode_binary_mask(self.mask)

        return record


class AnnotationManager:
    """Owns annotation state for one image at a time."""

    SNAPSHOT_VERSION = "1.0"

    def __init__(
        self,
        project_root: str | Path,
        project_name: str = "apple_demo",
        primary_threshold: float = 0.45,
        candidate_threshold: float = 0.30,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.project_name = self._safe_project_name(project_name)
        self.primary_threshold = float(primary_threshold)
        self.candidate_threshold = float(candidate_threshold)

        self.image_name: str | None = None
        self.image_width: int | None = None
        self.image_height: int | None = None
        self.source_image_path: str | None = None

        self.annotations: dict[int, AnnotationInstance] = {}
        self.next_instance_id: int = 1
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Project / image setup
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_project_name(name: str) -> str:
        text = str(name or "apple_demo").strip()
        allowed = []
        for char in text:
            if char.isalnum() or char in {"-", "_"}:
                allowed.append(char)
            elif char.isspace():
                allowed.append("_")
        result = "".join(allowed).strip("_")
        return result or "apple_demo"

    @property
    def workspace_root(self) -> Path:
        return self.project_root / "workspace" / "projects" / self.project_name

    @property
    def expected_mask_shape(self) -> tuple[int, int] | None:
        if self.image_width is None or self.image_height is None:
            return None
        return self.image_height, self.image_width

    def set_image(
        self,
        image_name: str,
        width: int,
        height: int,
        source_image_path: str | None = None,
        clear_annotations: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive.")

        self.image_name = Path(str(image_name or "image.png")).name
        self.image_width = int(width)
        self.image_height = int(height)
        self.source_image_path = source_image_path

        if clear_annotations:
            self.annotations.clear()
            self.next_instance_id = 1
            self.history.clear()

        self._log("set_image", None, {"image_name": self.image_name})

    def require_image(self) -> None:
        if self.image_name is None or self.expected_mask_shape is None:
            raise RuntimeError("No image is currently loaded in AnnotationManager.")

    # ------------------------------------------------------------------
    # Annotation creation / mutation
    # ------------------------------------------------------------------

    def add_instance(
        self,
        mask: np.ndarray,
        class_label: str,
        score: float | None = None,
        bbox_xyxy: Iterable[float] | None = None,
        source: str = "manual",
        tier: str = "manual",
        review_status: str = "corrected",
        notes: str = "",
    ) -> AnnotationInstance:
        self.require_image()
        mask = ensure_bool_mask(mask, self.expected_mask_shape)

        if int(mask.sum()) == 0:
            raise ValueError("Cannot add an empty mask.")

        annotation = AnnotationInstance(
            instance_id=self.next_instance_id,
            class_label=class_label,
            mask=mask.copy(),
            score=score,
            bbox_xyxy=normalize_bbox(bbox_xyxy, mask),
            source=source,
            tier=tier,
            review_status=review_status,
            notes=notes,
        )

        self.annotations[annotation.instance_id] = annotation
        self.next_instance_id += 1

        self._log(
            "add_instance",
            annotation.instance_id,
            {
                "class_label": annotation.class_label,
                "source": source,
                "tier": tier,
                "score": score,
            },
        )

        return annotation

    def add_auto_predictions(
        self,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_label: str,
        replace_existing: bool = False,
    ) -> list[AnnotationInstance]:
        self.require_image()

        masks = np.asarray(masks)
        boxes = np.asarray(boxes)
        scores = np.asarray(scores).reshape(-1)

        if masks.ndim == 2:
            masks = masks[None, ...]

        if len(masks) != len(boxes) or len(masks) != len(scores):
            raise ValueError(
                f"Prediction count mismatch: masks={len(masks)}, boxes={len(boxes)}, scores={len(scores)}"
            )

        if replace_existing:
            self.annotations.clear()
            self.next_instance_id = 1
            self.history.clear()

        created: list[AnnotationInstance] = []

        for mask, box, score in zip(masks, boxes, scores):
            score_value = float(score)
            if score_value < self.candidate_threshold:
                continue

            tier = "primary" if score_value >= self.primary_threshold else "candidate"
            created.append(
                self.add_instance(
                    mask=mask,
                    class_label=class_label,
                    score=score_value,
                    bbox_xyxy=box,
                    source="sam3_text",
                    tier=tier,
                    review_status="pending",
                )
            )

        self._log(
            "add_auto_predictions",
            None,
            {
                "count": len(created),
                "class_label": class_label,
                "primary_threshold": self.primary_threshold,
                "candidate_threshold": self.candidate_threshold,
            },
        )

        return created

    def get(self, instance_id: int, allow_deleted: bool = False) -> AnnotationInstance:
        instance_id = int(instance_id)
        if instance_id not in self.annotations:
            raise KeyError(f"Annotation instance #{instance_id} does not exist.")

        annotation = self.annotations[instance_id]
        if annotation.is_deleted and not allow_deleted:
            raise KeyError(f"Annotation instance #{instance_id} has been deleted.")
        return annotation

    def accept(self, instance_id: int) -> AnnotationInstance:
        annotation = self.get(instance_id)
        annotation.review_status = "accepted"
        annotation.updated_at = utc_now_iso()
        self._log("accept", annotation.instance_id, {})
        return annotation

    def delete(self, instance_id: int) -> AnnotationInstance:
        annotation = self.get(instance_id)
        annotation.review_status = "deleted"
        annotation.updated_at = utc_now_iso()
        self._log("delete", annotation.instance_id, {})
        return annotation

    def relabel(self, instance_id: int, class_label: str) -> AnnotationInstance:
        label = str(class_label).strip()
        if not label:
            raise ValueError("New class label cannot be empty.")

        annotation = self.get(instance_id)
        previous = annotation.class_label
        annotation.class_label = label
        annotation.review_status = "corrected"
        annotation.updated_at = utc_now_iso()
        self._log(
            "relabel",
            annotation.instance_id,
            {"from": previous, "to": label},
        )
        return annotation

    def replace_mask(
        self,
        instance_id: int,
        mask: np.ndarray,
        score: float | None = None,
        source: str = "sam3_click_refine",
    ) -> AnnotationInstance:
        annotation = self.get(instance_id)
        mask = ensure_bool_mask(mask, self.expected_mask_shape)

        if int(mask.sum()) == 0:
            raise ValueError("Refined mask is empty.")

        previous_area = annotation.area
        annotation.mask = mask.copy()
        annotation.bbox_xyxy = bbox_from_mask(mask)
        if score is not None:
            annotation.score = float(score)
        annotation.source = source
        annotation.review_status = "corrected"
        annotation.updated_at = utc_now_iso()

        self._log(
            "replace_mask",
            annotation.instance_id,
            {
                "previous_area": previous_area,
                "new_area": annotation.area,
                "score": annotation.score,
            },
        )
        return annotation

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def active_annotations(self, show_candidates: bool = True) -> list[AnnotationInstance]:
        rows = []
        for annotation in self.annotations.values():
            if annotation.is_deleted:
                continue
            if not show_candidates and annotation.tier == "candidate":
                continue
            rows.append(annotation)
        return sorted(rows, key=lambda item: item.instance_id)

    def find_instance_at(
        self,
        x: int,
        y: int,
        show_candidates: bool = True,
    ) -> AnnotationInstance | None:
        self.require_image()

        x = int(round(x))
        y = int(round(y))
        if x < 0 or y < 0 or x >= self.image_width or y >= self.image_height:
            return None

        hits = [
            ann
            for ann in self.active_annotations(show_candidates=show_candidates)
            if ann.mask[y, x]
        ]

        if not hits:
            return None

        # Prefer the smallest mask when masks overlap; it is usually the more
        # specific instance. Then prefer higher confidence.
        hits.sort(
            key=lambda ann: (
                ann.area,
                -(ann.score if ann.score is not None else -1.0),
                ann.instance_id,
            )
        )
        return hits[0]

    def table_rows(self, show_candidates: bool = True) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for ann in self.active_annotations(show_candidates=show_candidates):
            rows.append(
                [
                    ann.instance_id,
                    ann.class_label,
                    "" if ann.score is None else round(float(ann.score), 4),
                    ann.tier,
                    ann.review_status,
                    ann.area,
                    ann.source,
                ]
            )
        return rows

    def stats(self) -> dict[str, int]:
        active = [ann for ann in self.annotations.values() if not ann.is_deleted]
        return {
            "total_records": len(self.annotations),
            "active": len(active),
            "primary": sum(ann.tier == "primary" for ann in active),
            "candidate": sum(ann.tier == "candidate" for ann in active),
            "manual": sum(ann.tier == "manual" for ann in active),
            "pending": sum(ann.review_status == "pending" for ann in active),
            "accepted": sum(ann.review_status == "accepted" for ann in active),
            "corrected": sum(ann.review_status == "corrected" for ann in active),
            "deleted": sum(ann.review_status == "deleted" for ann in self.annotations.values()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_source_image(self, image: Image.Image) -> Path:
        self.require_image()
        output_dir = self.workspace_root / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / self.image_name
        image.convert("RGB").save(output_path)
        self.source_image_path = str(output_path.resolve())
        return output_path

    def _snapshot_locations(self, kind: str) -> tuple[Path, Path]:
        self.require_image()
        if kind not in {"auto", "reviewed"}:
            raise ValueError("kind must be 'auto' or 'reviewed'.")

        stem = Path(self.image_name).stem
        folder_name = "auto_annotations" if kind == "auto" else "reviewed_annotations"
        root = self.workspace_root / folder_name
        json_path = root / f"{stem}.json"
        mask_dir = root / "masks" / stem
        return json_path, mask_dir

    def save_snapshot(
        self,
        kind: str,
        image: Image.Image | None = None,
        include_deleted_masks: bool = False,
    ) -> Path:
        self.require_image()

        if image is not None:
            self.save_source_image(image)

        json_path, mask_dir = self._snapshot_locations(kind)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Recreate the mask folder so stale masks from earlier edits cannot
        # silently remain in the snapshot.
        if mask_dir.exists():
            shutil.rmtree(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []

        for annotation in sorted(self.annotations.values(), key=lambda item: item.instance_id):
            mask_relative = None

            if include_deleted_masks or not annotation.is_deleted:
                mask_name = f"instance_{annotation.instance_id:04d}.png"
                mask_path = mask_dir / mask_name
                Image.fromarray(annotation.mask.astype(np.uint8) * 255, mode="L").save(mask_path)
                mask_relative = str(mask_path.relative_to(self.workspace_root))

            records.append(
                annotation.to_record(
                    include_rle=False,
                    mask_path=mask_relative,
                )
            )

        payload = {
            "snapshot_version": self.SNAPSHOT_VERSION,
            "created_at": utc_now_iso(),
            "kind": kind,
            "project_name": self.project_name,
            "image": {
                "name": self.image_name,
                "width": self.image_width,
                "height": self.image_height,
                "source_image_path": self.source_image_path,
            },
            "thresholds": {
                "primary": self.primary_threshold,
                "candidate": self.candidate_threshold,
            },
            "stats": self.stats(),
            "annotations": records,
            "history": self.history,
        }

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

        return json_path

    def load_snapshot(self, json_path: str | Path) -> None:
        path = Path(json_path).resolve()
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        image_info = payload["image"]
        self.project_name = self._safe_project_name(payload.get("project_name", self.project_name))
        self.set_image(
            image_name=image_info["name"],
            width=int(image_info["width"]),
            height=int(image_info["height"]),
            source_image_path=image_info.get("source_image_path"),
            clear_annotations=True,
        )

        thresholds = payload.get("thresholds", {})
        self.primary_threshold = float(thresholds.get("primary", self.primary_threshold))
        self.candidate_threshold = float(thresholds.get("candidate", self.candidate_threshold))

        snapshot_root = path.parent
        if path.parent.name in {"auto_annotations", "reviewed_annotations"}:
            workspace_root = path.parent.parent
        else:
            workspace_root = self.workspace_root

        self.annotations.clear()
        max_id = 0

        for record in payload.get("annotations", []):
            mask_path_text = record.get("mask_path")
            if not mask_path_text:
                # Deleted records may intentionally have no persisted mask.
                continue

            mask_path = Path(mask_path_text)
            if not mask_path.is_absolute():
                # Paths are stored relative to the project workspace root.
                mask_path = workspace_root / mask_path

            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L")) > 0

            annotation = AnnotationInstance(
                instance_id=int(record["instance_id"]),
                class_label=record["class_label"],
                mask=mask,
                score=record.get("score"),
                bbox_xyxy=record.get("bbox_xyxy"),
                source=record.get("source", "loaded"),
                tier=record.get("tier", "manual"),
                review_status=record.get("review_status", "pending"),
                created_at=record.get("created_at", utc_now_iso()),
                updated_at=record.get("updated_at", utc_now_iso()),
                notes=record.get("notes", ""),
            )
            self.annotations[annotation.instance_id] = annotation
            max_id = max(max_id, annotation.instance_id)

        self.next_instance_id = max_id + 1
        self.history = list(payload.get("history", []))
        self._log("load_snapshot", None, {"path": str(path)})

    # ------------------------------------------------------------------
    # COCO export for current reviewed image
    # ------------------------------------------------------------------

    def export_coco(self, output_path: str | Path | None = None) -> Path:
        self.require_image()

        if output_path is None:
            export_dir = self.workspace_root / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            output_path = export_dir / f"{Path(self.image_name).stem}_coco.json"
        else:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = self.workspace_root / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

        active = self.active_annotations(show_candidates=True)
        labels = sorted({ann.class_label for ann in active})
        category_ids = {label: index + 1 for index, label in enumerate(labels)}

        coco_annotations: list[dict[str, Any]] = []
        annotation_id = 1

        for ann in active:
            rle = encode_binary_mask(ann.mask)
            bbox_xywh = bbox_xyxy_to_xywh(ann.bbox_xyxy)

            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": 1,
                    "category_id": category_ids[ann.class_label],
                    "segmentation": rle,
                    "area": ann.area,
                    "bbox": bbox_xywh,
                    "iscrowd": 0,
                    "attributes": {
                        "instance_id": ann.instance_id,
                        "score": ann.score,
                        "source": ann.source,
                        "tier": ann.tier,
                        "review_status": ann.review_status,
                    },
                }
            )
            annotation_id += 1

        payload = {
            "info": {
                "description": "AgriAnnotationSystem reviewed annotation export",
                "date_created": utc_now_iso(),
                "project_name": self.project_name,
            },
            "images": [
                {
                    "id": 1,
                    "file_name": self.image_name,
                    "width": self.image_width,
                    "height": self.image_height,
                }
            ],
            "categories": [
                {"id": category_ids[label], "name": label, "supercategory": "crop"}
                for label in labels
            ],
            "annotations": coco_annotations,
        }

        output_path = Path(output_path).resolve()
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

        self._log("export_coco", None, {"path": str(output_path), "instances": len(active)})
        return output_path

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _log(self, action: str, instance_id: int | None, details: dict[str, Any]) -> None:
        self.history.append(
            {
                "time": utc_now_iso(),
                "action": action,
                "instance_id": instance_id,
                "details": details,
            }
        )


__all__ = [
    "AnnotationInstance",
    "AnnotationManager",
    "bbox_from_mask",
    "bbox_xyxy_to_xywh",
    "encode_binary_mask",
    "decode_binary_mask",
]
