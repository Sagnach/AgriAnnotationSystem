from pathlib import Path
from collections import Counter
from datetime import datetime
from PIL import Image, ImageDraw

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import tarfile
import zipfile

import numpy as np


# ============================================================
# MinneApple Dataset Audit
# ============================================================
#
# Project:
#   AgriAnnotationSystem
#
# Purpose:
#   1. Locate downloaded MinneApple archives
#   2. Check archive integrity
#   3. Record archive contents
#   4. Safely extract archives
#   5. Automatically discover images/ + masks/ dataset roots
#   6. Verify image-mask pairing
#   7. Verify image/mask dimensions
#   8. Count apple instances
#   9. Detect corrupt files
#  10. Generate visual audit previews
#  11. Save CSV and JSON audit reports
#
# This script DOES NOT:
#   - train a model
#   - modify original archives
#   - convert the dataset
#   - create train/val/test splits
#
# ============================================================


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


EXPECTED_ARCHIVES = {
    "detection": {
        "filename": "detection.tar.gz",
        "patterns": [
            "*detection*.tar.gz",
            "*detection*.tgz",
        ],
    },
    "test_data": {
        "filename": "test_data.zip",
        "patterns": [
            "*test*data*.zip",
            "*test*.zip",
        ],
    },
}


# ============================================================
# General helpers
# ============================================================


def human_bytes(num_bytes):
    value = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} PB"


def relative_path(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Archive discovery
# ============================================================


def locate_archive(raw_root, archive_dir, label, config):
    exact_candidates = [
        archive_dir / config["filename"],
        raw_root / config["filename"],
    ]

    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate

    fuzzy_candidates = []

    for pattern in config["patterns"]:
        fuzzy_candidates.extend(
            raw_root.rglob(pattern)
        )

    fuzzy_candidates = sorted(
        {
            path.resolve()
            for path in fuzzy_candidates
            if path.is_file()
        }
    )

    if len(fuzzy_candidates) == 1:
        return Path(fuzzy_candidates[0])

    if len(fuzzy_candidates) > 1:
        print()
        print(
            f"[ERROR] Multiple candidate archives found "
            f"for '{label}':"
        )

        for path in fuzzy_candidates:
            print(f"        {path}")

        return None

    return None


# ============================================================
# Archive inspection
# ============================================================


def inspect_tar_archive(archive_path):
    rows = []
    extension_counter = Counter()
    total_size = 0
    file_count = 0
    directory_count = 0

    with tarfile.open(
        archive_path,
        mode="r:*",
    ) as tar:
        members = tar.getmembers()

        for member in members:
            if member.isdir():
                directory_count += 1
                item_type = "directory"

            elif member.isfile():
                file_count += 1
                total_size += member.size
                item_type = "file"

                suffix = Path(member.name).suffix.lower()

                if suffix:
                    extension_counter[suffix] += 1

            elif member.issym():
                item_type = "symlink"

            elif member.islnk():
                item_type = "hardlink"

            else:
                item_type = "other"

            rows.append(
                {
                    "archive": archive_path.name,
                    "member": member.name,
                    "type": item_type,
                    "size_bytes": member.size,
                }
            )

    summary = {
        "archive": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_size": human_bytes(
            archive_path.stat().st_size
        ),
        "member_files": file_count,
        "member_directories": directory_count,
        "uncompressed_bytes": total_size,
        "uncompressed_size": human_bytes(total_size),
        "extensions": dict(
            sorted(extension_counter.items())
        ),
        "integrity": "PASS",
    }

    return summary, rows


def inspect_zip_archive(archive_path):
    rows = []
    extension_counter = Counter()
    total_size = 0
    file_count = 0
    directory_count = 0

    with zipfile.ZipFile(
        archive_path,
        mode="r",
    ) as archive:

        bad_file = archive.testzip()

        if bad_file is not None:
            raise RuntimeError(
                f"ZIP integrity check failed at: {bad_file}"
            )

        for info in archive.infolist():

            if info.is_dir():
                directory_count += 1
                item_type = "directory"

            else:
                file_count += 1
                total_size += info.file_size
                item_type = "file"

                suffix = Path(info.filename).suffix.lower()

                if suffix:
                    extension_counter[suffix] += 1

            rows.append(
                {
                    "archive": archive_path.name,
                    "member": info.filename,
                    "type": item_type,
                    "size_bytes": info.file_size,
                }
            )

    summary = {
        "archive": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_size": human_bytes(
            archive_path.stat().st_size
        ),
        "member_files": file_count,
        "member_directories": directory_count,
        "uncompressed_bytes": total_size,
        "uncompressed_size": human_bytes(total_size),
        "extensions": dict(
            sorted(extension_counter.items())
        ),
        "integrity": "PASS",
    }

    return summary, rows


def inspect_archive(archive_path):
    lower_name = archive_path.name.lower()

    if (
        lower_name.endswith(".tar.gz")
        or lower_name.endswith(".tgz")
        or lower_name.endswith(".tar")
    ):
        return inspect_tar_archive(
            archive_path
        )

    if lower_name.endswith(".zip"):
        return inspect_zip_archive(
            archive_path
        )

    raise RuntimeError(
        f"Unsupported archive format: {archive_path}"
    )


# ============================================================
# Safe archive extraction
# ============================================================


def ensure_safe_target(destination, target):
    destination = destination.resolve()
    target = target.resolve()

    try:
        target.relative_to(destination)
    except ValueError:
        raise RuntimeError(
            f"Unsafe archive path detected: {target}"
        )


def safe_extract_tar(archive_path, destination):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        archive_path,
        mode="r:*",
    ) as tar:

        safe_members = []

        for member in tar.getmembers():

            target = destination / member.name

            ensure_safe_target(
                destination,
                target,
            )

            if member.issym() or member.islnk():
                raise RuntimeError(
                    "Archive contains a symbolic/hard link: "
                    f"{member.name}"
                )

            safe_members.append(member)

        try:
            tar.extractall(
                destination,
                members=safe_members,
                filter="data",
            )

        except TypeError:
            # Compatibility fallback for Python versions
            # without the filter argument.
            tar.extractall(
                destination,
                members=safe_members,
            )


def zip_entry_is_symlink(info):
    mode = (
        info.external_attr >> 16
    ) & 0o170000

    return mode == 0o120000


def safe_extract_zip(archive_path, destination):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        archive_path,
        mode="r",
    ) as archive:

        for info in archive.infolist():

            target = (
                destination / info.filename
            )

            ensure_safe_target(
                destination,
                target,
            )

            if zip_entry_is_symlink(info):
                raise RuntimeError(
                    "ZIP contains a symbolic link: "
                    f"{info.filename}"
                )

        archive.extractall(destination)


def archive_marker_matches(
    marker_path,
    archive_path,
):
    if not marker_path.exists():
        return False

    try:
        data = json.loads(
            marker_path.read_text(
                encoding="utf-8"
            )
        )

        return (
            data.get("archive_name")
            == archive_path.name
            and data.get("archive_size")
            == archive_path.stat().st_size
        )

    except Exception:
        return False


def extract_archive(
    archive_path,
    destination,
    force=False,
):
    marker = destination / ".extraction_complete.json"

    if (
        destination.exists()
        and archive_marker_matches(
            marker,
            archive_path,
        )
        and not force
    ):
        print(
            f"  [SKIP] Already extracted: "
            f"{destination}"
        )
        return

    if destination.exists():

        existing_items = list(
            destination.iterdir()
        )

        if existing_items:

            if force:
                print(
                    f"  [INFO] Removing previous "
                    f"extraction: {destination}"
                )

                shutil.rmtree(destination)

            else:
                raise RuntimeError(
                    "Extraction directory already contains "
                    "data but has no valid completion marker:\n"
                    f"{destination}\n"
                    "Run again with --force-extract if "
                    "you want to recreate it."
                )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"  Extracting {archive_path.name} ..."
    )

    lower_name = archive_path.name.lower()

    if (
        lower_name.endswith(".tar.gz")
        or lower_name.endswith(".tgz")
        or lower_name.endswith(".tar")
    ):
        safe_extract_tar(
            archive_path,
            destination,
        )

    elif lower_name.endswith(".zip"):
        safe_extract_zip(
            archive_path,
            destination,
        )

    else:
        raise RuntimeError(
            f"Unsupported archive: {archive_path}"
        )

    marker_data = {
        "archive_name": archive_path.name,
        "archive_size": archive_path.stat().st_size,
        "completed_at": datetime.now().isoformat(),
    }

    write_json(
        marker,
        marker_data,
    )

    print(
        f"  [OK] Extracted to: {destination}"
    )


# ============================================================
# Dataset root discovery
# ============================================================


def discover_dataset_roots(extracted_root):
    roots = []

    for images_dir in extracted_root.rglob("images"):

        if not images_dir.is_dir():
            continue

        parent = images_dir.parent
        masks_dir = parent / "masks"

        if masks_dir.is_dir():
            roots.append(parent)

    return sorted(
        set(roots),
        key=lambda path: str(path).lower(),
    )


def list_image_files(directory):
    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=lambda path: path.name.lower(),
    )


def map_files_by_stem(files):
    result = {}
    duplicates = []

    for path in files:
        key = path.stem.lower()

        if key in result:
            duplicates.append(
                (
                    key,
                    result[key],
                    path,
                )
            )
        else:
            result[key] = path

    return result, duplicates


# ============================================================
# Mask analysis
# ============================================================


def analyze_mask_array(mask_array):
    if mask_array.ndim == 2:

        unique_ids = np.unique(mask_array)

        has_background = bool(
            np.any(unique_ids == 0)
        )

        instance_ids = unique_ids[
            unique_ids != 0
        ]

        return {
            "instance_count": int(
                len(instance_ids)
            ),
            "has_background": has_background,
            "mask_kind": "indexed_2d",
            "unique_values": int(
                len(unique_ids)
            ),
        }

    if mask_array.ndim == 3:

        # Ignore alpha channel if present.
        color_array = mask_array[..., :3]

        pixels = color_array.reshape(
            -1,
            color_array.shape[-1],
        )

        unique_colors = np.unique(
            pixels,
            axis=0,
        )

        background_rows = np.all(
            unique_colors == 0,
            axis=1,
        )

        has_background = bool(
            np.any(background_rows)
        )

        instance_count = int(
            len(unique_colors)
            - np.sum(background_rows)
        )

        return {
            "instance_count": instance_count,
            "has_background": has_background,
            "mask_kind": "color_3d",
            "unique_values": int(
                len(unique_colors)
            ),
        }

    return {
        "instance_count": 0,
        "has_background": False,
        "mask_kind": (
            f"unsupported_{mask_array.ndim}d"
        ),
        "unique_values": 0,
    }


# ============================================================
# Split hint
# ============================================================


def infer_split_hint(path):
    text = str(path).lower()

    if "test" in text:
        return "test"

    if "val" in text:
        return "val"

    if "train" in text:
        return "train"

    return "unknown"


# ============================================================
# Dataset audit
# ============================================================


def audit_dataset_root(
    dataset_root,
    project_root,
):
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"

    image_files = list_image_files(
        images_dir
    )

    mask_files = list_image_files(
        masks_dir
    )

    image_map, image_duplicates = (
        map_files_by_stem(image_files)
    )

    mask_map, mask_duplicates = (
        map_files_by_stem(mask_files)
    )

    image_stems = set(image_map.keys())
    mask_stems = set(mask_map.keys())

    paired_stems = sorted(
        image_stems & mask_stems
    )

    images_without_masks = sorted(
        image_stems - mask_stems
    )

    masks_without_images = sorted(
        mask_stems - image_stems
    )

    pair_rows = []

    corrupt_images = []
    corrupt_masks = []
    size_mismatches = []
    empty_masks = []
    missing_background = []
    unsupported_masks = []

    instance_counts = []
    image_size_counter = Counter()
    mask_mode_counter = Counter()

    for index, stem in enumerate(
        paired_stems,
        start=1,
    ):

        image_path = image_map[stem]
        mask_path = mask_map[stem]

        row = {
            "dataset_root": relative_path(
                dataset_root,
                project_root,
            ),
            "split_hint": infer_split_hint(
                dataset_root
            ),
            "stem": stem,
            "image_path": relative_path(
                image_path,
                project_root,
            ),
            "mask_path": relative_path(
                mask_path,
                project_root,
            ),
            "image_width": "",
            "image_height": "",
            "mask_width": "",
            "mask_height": "",
            "image_mode": "",
            "mask_mode": "",
            "mask_kind": "",
            "instance_count": "",
            "has_background": "",
            "size_match": "",
            "status": "PASS",
            "error": "",
        }

        try:
            with Image.open(
                image_path
            ) as image:

                image.load()

                image_width, image_height = (
                    image.size
                )

                image_mode = image.mode

            row["image_width"] = image_width
            row["image_height"] = image_height
            row["image_mode"] = image_mode

            image_size_counter[
                f"{image_width}x{image_height}"
            ] += 1

        except Exception as error:
            row["status"] = "FAIL"
            row["error"] = (
                f"Image error: {error}"
            )

            corrupt_images.append(
                str(image_path)
            )

            pair_rows.append(row)
            continue

        try:
            with Image.open(
                mask_path
            ) as mask:

                mask.load()

                mask_width, mask_height = (
                    mask.size
                )

                mask_mode = mask.mode

                mask_array = np.array(
                    mask
                )

            row["mask_width"] = mask_width
            row["mask_height"] = mask_height
            row["mask_mode"] = mask_mode

            mask_mode_counter[
                mask_mode
            ] += 1

            mask_info = analyze_mask_array(
                mask_array
            )

            instance_count = (
                mask_info[
                    "instance_count"
                ]
            )

            has_background = (
                mask_info[
                    "has_background"
                ]
            )

            mask_kind = (
                mask_info[
                    "mask_kind"
                ]
            )

            row["mask_kind"] = mask_kind
            row["instance_count"] = (
                instance_count
            )
            row["has_background"] = (
                has_background
            )

            instance_counts.append(
                instance_count
            )

            if instance_count == 0:
                empty_masks.append(
                    str(mask_path)
                )

            if not has_background:
                missing_background.append(
                    str(mask_path)
                )

            if mask_kind.startswith(
                "unsupported"
            ):
                unsupported_masks.append(
                    str(mask_path)
                )

        except Exception as error:
            row["status"] = "FAIL"

            if row["error"]:
                row["error"] += " | "

            row["error"] += (
                f"Mask error: {error}"
            )

            corrupt_masks.append(
                str(mask_path)
            )

            pair_rows.append(row)
            continue

        size_match = (
            image_width == mask_width
            and image_height == mask_height
        )

        row["size_match"] = size_match

        if not size_match:
            row["status"] = "FAIL"

            row["error"] = (
                "Image/mask dimension mismatch"
            )

            size_mismatches.append(
                {
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "image_size": [
                        image_width,
                        image_height,
                    ],
                    "mask_size": [
                        mask_width,
                        mask_height,
                    ],
                }
            )

        pair_rows.append(row)

        if (
            index % 100 == 0
            or index == len(paired_stems)
        ):
            print(
                f"      Audited "
                f"{index}/{len(paired_stems)} pairs"
            )

    if instance_counts:
        instance_statistics = {
            "total_instances": int(
                sum(instance_counts)
            ),
            "mean_instances_per_image": round(
                statistics.mean(
                    instance_counts
                ),
                3,
            ),
            "median_instances_per_image": round(
                statistics.median(
                    instance_counts
                ),
                3,
            ),
            "min_instances_per_image": int(
                min(instance_counts)
            ),
            "max_instances_per_image": int(
                max(instance_counts)
            ),
        }

    else:
        instance_statistics = {
            "total_instances": 0,
            "mean_instances_per_image": 0,
            "median_instances_per_image": 0,
            "min_instances_per_image": 0,
            "max_instances_per_image": 0,
        }

    summary = {
        "dataset_root": relative_path(
            dataset_root,
            project_root,
        ),
        "split_hint": infer_split_hint(
            dataset_root
        ),
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "paired_count": len(paired_stems),

        "images_without_masks_count":
            len(images_without_masks),

        "masks_without_images_count":
            len(masks_without_images),

        "duplicate_image_stems":
            len(image_duplicates),

        "duplicate_mask_stems":
            len(mask_duplicates),

        "corrupt_images":
            len(corrupt_images),

        "corrupt_masks":
            len(corrupt_masks),

        "size_mismatches":
            len(size_mismatches),

        "empty_masks":
            len(empty_masks),

        "masks_without_background":
            len(missing_background),

        "unsupported_masks":
            len(unsupported_masks),

        "image_sizes":
            dict(image_size_counter),

        "mask_modes":
            dict(mask_mode_counter),

        **instance_statistics,

        "images_without_masks":
            images_without_masks,

        "masks_without_images":
            masks_without_images,

        "corrupt_image_files":
            corrupt_images,

        "corrupt_mask_files":
            corrupt_masks,

        "size_mismatch_details":
            size_mismatches,

        "empty_mask_files":
            empty_masks,

        "missing_background_files":
            missing_background,

        "unsupported_mask_files":
            unsupported_masks,
    }

    return (
        summary,
        pair_rows,
        [
            (
                image_map[stem],
                mask_map[stem],
            )
            for stem in paired_stems
        ],
    )


# ============================================================
# Preview generation
# ============================================================


def overlay_indexed_mask(
    image_array,
    mask_array,
    rng,
):
    result = image_array.astype(
        np.float32
    ).copy()

    instance_ids = np.unique(
        mask_array
    )

    instance_ids = instance_ids[
        instance_ids != 0
    ]

    alpha = 0.45

    for instance_id in instance_ids:

        region = (
            mask_array == instance_id
        )

        color = rng.integers(
            low=40,
            high=256,
            size=3,
        ).astype(np.float32)

        result[region] = (
            (1.0 - alpha)
            * result[region]
            + alpha
            * color
        )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def overlay_color_mask(
    image_array,
    mask_array,
    rng,
):
    result = image_array.astype(
        np.float32
    ).copy()

    colors = np.unique(
        mask_array[..., :3].reshape(
            -1,
            3,
        ),
        axis=0,
    )

    alpha = 0.45

    for mask_color in colors:

        if np.all(mask_color == 0):
            continue

        region = np.all(
            mask_array[..., :3]
            == mask_color,
            axis=-1,
        )

        display_color = rng.integers(
            low=40,
            high=256,
            size=3,
        ).astype(np.float32)

        result[region] = (
            (1.0 - alpha)
            * result[region]
            + alpha
            * display_color
        )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def create_preview(
    image_path,
    mask_path,
    output_path,
    seed,
):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_array = np.array(image)

    with Image.open(mask_path) as mask:
        mask_array = np.array(mask)

    rng = np.random.default_rng(seed)

    mask_info = analyze_mask_array(
        mask_array
    )

    if mask_array.ndim == 2:
        overlay_array = (
            overlay_indexed_mask(
                image_array,
                mask_array,
                rng,
            )
        )

    elif mask_array.ndim == 3:
        overlay_array = (
            overlay_color_mask(
                image_array,
                mask_array,
                rng,
            )
        )

    else:
        return False

    original = Image.fromarray(
        image_array
    )

    overlay = Image.fromarray(
        overlay_array
    )

    width, height = original.size

    canvas = Image.new(
        "RGB",
        (
            width * 2,
            height + 40,
        ),
        "white",
    )

    canvas.paste(
        original,
        (0, 40),
    )

    canvas.paste(
        overlay,
        (width, 40),
    )

    draw = ImageDraw.Draw(canvas)

    title = (
        f"{image_path.name} | "
        f"instances={mask_info['instance_count']} | "
        "LEFT=original | RIGHT=GT overlay"
    )

    draw.text(
        (10, 10),
        title,
        fill="black",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        quality=92,
    )

    return True


# ============================================================
# Overall status
# ============================================================


def determine_status(
    archive_errors,
    dataset_summaries,
):
    errors = list(archive_errors)
    warnings = []

    if not dataset_summaries:
        errors.append(
            "No dataset root with sibling "
            "images/ and masks/ was found."
        )

    for summary in dataset_summaries:

        root_name = summary[
            "dataset_root"
        ]

        if (
            summary[
                "images_without_masks_count"
            ]
            > 0
        ):
            errors.append(
                f"{root_name}: images without masks"
            )

        if (
            summary[
                "masks_without_images_count"
            ]
            > 0
        ):
            errors.append(
                f"{root_name}: masks without images"
            )

        if summary[
            "corrupt_images"
        ] > 0:
            errors.append(
                f"{root_name}: corrupt images"
            )

        if summary[
            "corrupt_masks"
        ] > 0:
            errors.append(
                f"{root_name}: corrupt masks"
            )

        if summary[
            "size_mismatches"
        ] > 0:
            errors.append(
                f"{root_name}: image/mask "
                "dimension mismatches"
            )

        if summary[
            "duplicate_image_stems"
        ] > 0:
            errors.append(
                f"{root_name}: duplicate "
                "image stems"
            )

        if summary[
            "duplicate_mask_stems"
        ] > 0:
            errors.append(
                f"{root_name}: duplicate "
                "mask stems"
            )

        if summary[
            "empty_masks"
        ] > 0:
            warnings.append(
                f"{root_name}: empty masks found"
            )

        if summary[
            "masks_without_background"
        ] > 0:
            warnings.append(
                f"{root_name}: masks without "
                "background ID 0 found"
            )

        if summary[
            "unsupported_masks"
        ] > 0:
            errors.append(
                f"{root_name}: unsupported "
                "mask dimensionality"
            )

    if errors:
        status = "FAIL"

    elif warnings:
        status = "PASS_WITH_WARNINGS"

    else:
        status = "PASS"

    return status, errors, warnings


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit the downloaded MinneApple "
            "dataset archives."
        )
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Optional project root override. "
            "Normally auto-detected."
        ),
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help=(
            "Inspect archives only; do not extract."
        ),
    )

    parser.add_argument(
        "--force-extract",
        action="store_true",
        help=(
            "Delete and recreate previous "
            "extraction directories."
        ),
    )

    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help=(
            "Skip SHA256 checksum calculation."
        ),
    )

    parser.add_argument(
        "--previews",
        type=int,
        default=8,
        help=(
            "Number of random image/mask previews "
            "to generate."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve project root
    # --------------------------------------------------------

    if args.project_root is None:

        # Script expected at:
        # AgriAnnotationSystem/scripts/01_audit_minneapple.py
        project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

    else:
        project_root = Path(
            args.project_root
        ).resolve()

    raw_root = (
        project_root
        / "data"
        / "raw"
        / "minneapple"
    )

    archive_dir = (
        raw_root
        / "_archives"
    )

    extracted_root = (
        raw_root
        / "_extracted"
    )

    report_root = (
        project_root
        / "outputs"
        / "reports"
        / "minneapple_audit"
    )

    preview_root = (
        report_root
        / "previews"
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print(
        "MINNEAPPLE DATASET AUDIT"
    )
    print("=" * 78)

    print(
        f"Project root : {project_root}"
    )

    print(
        f"Raw root     : {raw_root}"
    )

    print(
        f"Archive dir  : {archive_dir}"
    )

    print(
        f"Report dir   : {report_root}"
    )

    print()

    # --------------------------------------------------------
    # Stage 1: Locate archives
    # --------------------------------------------------------

    print(
        "[1/7] Locating MinneApple archives..."
    )

    archives = {}
    archive_errors = []

    for label, config in (
        EXPECTED_ARCHIVES.items()
    ):

        archive_path = locate_archive(
            raw_root,
            archive_dir,
            label,
            config,
        )

        if archive_path is None:

            message = (
                f"Missing archive: "
                f"{config['filename']}"
            )

            archive_errors.append(
                message
            )

            print(
                f"  [FAIL] {message}"
            )

        else:
            archives[label] = (
                archive_path
            )

            print(
                f"  [OK] {label:<12}: "
                f"{archive_path}"
            )

    if not archives:
        print()
        print(
            "[FATAL] No usable archives found."
        )

        print(
            "Expected files under:"
        )

        print(
            archive_dir
        )

        raise SystemExit(2)

    # --------------------------------------------------------
    # Stage 2: Archive integrity
    # --------------------------------------------------------

    print()
    print(
        "[2/7] Checking archive integrity..."
    )

    archive_summaries = {}
    archive_inventory_rows = []

    valid_archives = {}

    for label, archive_path in (
        archives.items()
    ):

        print()
        print(
            f"  Inspecting: "
            f"{archive_path.name}"
        )

        try:
            summary, rows = (
                inspect_archive(
                    archive_path
                )
            )

            if not args.skip_hash:

                print(
                    "    Calculating SHA256..."
                )

                summary[
                    "sha256"
                ] = sha256_file(
                    archive_path
                )

            else:
                summary[
                    "sha256"
                ] = "SKIPPED"

            archive_summaries[
                label
            ] = summary

            archive_inventory_rows.extend(
                rows
            )

            valid_archives[
                label
            ] = archive_path

            print(
                "    Integrity       : PASS"
            )

            print(
                "    Archive size    : "
                f"{summary['archive_size']}"
            )

            print(
                "    Files           : "
                f"{summary['member_files']}"
            )

            print(
                "    Uncompressed    : "
                f"{summary['uncompressed_size']}"
            )

            print(
                "    File extensions : "
                f"{summary['extensions']}"
            )

        except Exception as error:

            message = (
                f"{archive_path.name}: "
                f"{error}"
            )

            archive_errors.append(
                message
            )

            print(
                f"    [FAIL] {error}"
            )

    inventory_csv = (
        report_root
        / "archive_inventory.csv"
    )

    write_csv(
        inventory_csv,
        archive_inventory_rows,
        [
            "archive",
            "member",
            "type",
            "size_bytes",
        ],
    )

    # --------------------------------------------------------
    # Stage 3: Extract archives
    # --------------------------------------------------------

    print()
    print(
        "[3/7] Extracting archives..."
    )

    if args.no_extract:

        print(
            "  [SKIP] Extraction disabled "
            "by --no-extract"
        )

    else:

        for label, archive_path in (
            valid_archives.items()
        ):

            destination = (
                extracted_root
                / label
            )

            try:
                extract_archive(
                    archive_path,
                    destination,
                    force=args.force_extract,
                )

            except Exception as error:

                message = (
                    f"Extraction failed for "
                    f"{archive_path.name}: "
                    f"{error}"
                )

                archive_errors.append(
                    message
                )

                print(
                    f"  [FAIL] {message}"
                )

    # --------------------------------------------------------
    # Stage 4: Discover dataset roots
    # --------------------------------------------------------

    print()
    print(
        "[4/7] Discovering image/mask datasets..."
    )

    dataset_roots = []

    if extracted_root.exists():

        dataset_roots = (
            discover_dataset_roots(
                extracted_root
            )
        )

    if not dataset_roots:

        print(
            "  [FAIL] No directory containing "
            "sibling images/ and masks/ "
            "was discovered."
        )

    else:

        for index, root in enumerate(
            dataset_roots,
            start=1,
        ):

            print(
                f"  [{index}] "
                f"{relative_path(root, project_root)}"
            )

    # --------------------------------------------------------
    # Stage 5: Audit image-mask pairs
    # --------------------------------------------------------

    print()
    print(
        "[5/7] Auditing image-mask pairs..."
    )

    dataset_summaries = []
    all_pair_rows = []
    all_preview_pairs = []

    for index, dataset_root in enumerate(
        dataset_roots,
        start=1,
    ):

        print()
        print(
            f"  Dataset root {index}:"
        )

        print(
            f"    {dataset_root}"
        )

        (
            summary,
            pair_rows,
            preview_pairs,
        ) = audit_dataset_root(
            dataset_root,
            project_root,
        )

        dataset_summaries.append(
            summary
        )

        all_pair_rows.extend(
            pair_rows
        )

        all_preview_pairs.extend(
            preview_pairs
        )

        print(
            "    Images                : "
            f"{summary['image_count']}"
        )

        print(
            "    Masks                 : "
            f"{summary['mask_count']}"
        )

        print(
            "    Paired                : "
            f"{summary['paired_count']}"
        )

        print(
            "    Apple instances       : "
            f"{summary['total_instances']}"
        )

        print(
            "    Mean instances/image  : "
            f"{summary['mean_instances_per_image']}"
        )

        print(
            "    Images without masks  : "
            f"{summary['images_without_masks_count']}"
        )

        print(
            "    Masks without images  : "
            f"{summary['masks_without_images_count']}"
        )

        print(
            "    Corrupt images        : "
            f"{summary['corrupt_images']}"
        )

        print(
            "    Corrupt masks         : "
            f"{summary['corrupt_masks']}"
        )

        print(
            "    Size mismatches       : "
            f"{summary['size_mismatches']}"
        )

        print(
            "    Empty masks           : "
            f"{summary['empty_masks']}"
        )

        print(
            "    Image sizes           : "
            f"{summary['image_sizes']}"
        )

        print(
            "    Mask modes            : "
            f"{summary['mask_modes']}"
        )

    pairs_csv = (
        report_root
        / "image_mask_audit.csv"
    )

    write_csv(
        pairs_csv,
        all_pair_rows,
        [
            "dataset_root",
            "split_hint",
            "stem",
            "image_path",
            "mask_path",
            "image_width",
            "image_height",
            "mask_width",
            "mask_height",
            "image_mode",
            "mask_mode",
            "mask_kind",
            "instance_count",
            "has_background",
            "size_match",
            "status",
            "error",
        ],
    )

    # --------------------------------------------------------
    # Stage 6: Generate previews
    # --------------------------------------------------------

    print()
    print(
        "[6/7] Generating visual previews..."
    )

    preview_files = []

    if all_preview_pairs:

        random_generator = random.Random(
            args.seed
        )

        sample_count = min(
            max(args.previews, 0),
            len(all_preview_pairs),
        )

        selected_pairs = (
            random_generator.sample(
                all_preview_pairs,
                sample_count,
            )
            if sample_count > 0
            else []
        )

        for index, (
            image_path,
            mask_path,
        ) in enumerate(
            selected_pairs,
            start=1,
        ):

            output_path = (
                preview_root
                / (
                    f"preview_{index:02d}_"
                    f"{image_path.stem}.jpg"
                )
            )

            try:
                success = create_preview(
                    image_path,
                    mask_path,
                    output_path,
                    seed=args.seed + index,
                )

                if success:
                    preview_files.append(
                        relative_path(
                            output_path,
                            project_root,
                        )
                    )

                    print(
                        f"  [OK] "
                        f"{output_path.name}"
                    )

                else:
                    print(
                        f"  [WARN] Could not "
                        f"preview "
                        f"{image_path.name}"
                    )

            except Exception as error:

                print(
                    f"  [WARN] Preview failed "
                    f"for {image_path.name}: "
                    f"{error}"
                )

    else:
        print(
            "  [SKIP] No valid pairs "
            "available for preview."
        )

    # --------------------------------------------------------
    # Stage 7: Final summary
    # --------------------------------------------------------

    print()
    print(
        "[7/7] Building final audit report..."
    )

    (
        final_status,
        final_errors,
        final_warnings,
    ) = determine_status(
        archive_errors,
        dataset_summaries,
    )

    total_images = sum(
        summary["image_count"]
        for summary in dataset_summaries
    )

    total_masks = sum(
        summary["mask_count"]
        for summary in dataset_summaries
    )

    total_pairs = sum(
        summary["paired_count"]
        for summary in dataset_summaries
    )

    total_instances = sum(
        summary["total_instances"]
        for summary in dataset_summaries
    )

    audit_summary = {
        "audit_version": "1.0",
        "created_at": (
            datetime.now().isoformat()
        ),
        "status": final_status,

        "project_root": str(
            project_root
        ),

        "raw_root": str(
            raw_root
        ),

        "archive_directory": str(
            archive_dir
        ),

        "extracted_root": str(
            extracted_root
        ),

        "archive_summaries":
            archive_summaries,

        "dataset_roots_found":
            len(dataset_summaries),

        "totals": {
            "images": total_images,
            "masks": total_masks,
            "paired_images_masks":
                total_pairs,
            "apple_instances":
                total_instances,
        },

        "datasets":
            dataset_summaries,

        "preview_files":
            preview_files,

        "errors":
            final_errors,

        "warnings":
            final_warnings,
    }

    summary_json = (
        report_root
        / "audit_summary.json"
    )

    write_json(
        summary_json,
        audit_summary,
    )

    print()
    print("=" * 78)
    print(
        "MINNEAPPLE AUDIT COMPLETE"
    )
    print("=" * 78)

    print(
        f"STATUS                 : "
        f"{final_status}"
    )

    print(
        f"DATASET ROOTS          : "
        f"{len(dataset_summaries)}"
    )

    print(
        f"TOTAL IMAGES           : "
        f"{total_images}"
    )

    print(
        f"TOTAL MASKS            : "
        f"{total_masks}"
    )

    print(
        f"PAIRED IMAGE/MASK      : "
        f"{total_pairs}"
    )

    print(
        f"TOTAL APPLE INSTANCES  : "
        f"{total_instances}"
    )

    print(
        f"ERRORS                 : "
        f"{len(final_errors)}"
    )

    print(
        f"WARNINGS               : "
        f"{len(final_warnings)}"
    )

    print()
    print(
        f"SUMMARY JSON           : "
        f"{summary_json}"
    )

    print(
        f"PAIR AUDIT CSV         : "
        f"{pairs_csv}"
    )

    print(
        f"ARCHIVE INVENTORY CSV  : "
        f"{inventory_csv}"
    )

    print(
        f"PREVIEWS               : "
        f"{preview_root}"
    )

    if final_errors:

        print()
        print(
            "Errors:"
        )

        for error in final_errors:
            print(
                f"  - {error}"
            )

    if final_warnings:

        print()
        print(
            "Warnings:"
        )

        for warning in final_warnings:
            print(
                f"  - {warning}"
            )

    print("=" * 78)

    if final_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()