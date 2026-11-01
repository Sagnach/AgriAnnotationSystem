from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse
import csv
import json
import os
import random
import shutil
import sys


# ============================================================
# MinneApple Dataset Preparation
# ============================================================
#
# Purpose:
#   1. Read the PASS result from 01_audit_minneapple.py
#   2. Load audited image/mask pairs
#   3. Preserve the official MinneApple test set unchanged
#   4. Split the official training set into train/validation
#      without mixing images from the same acquisition group
#   5. Build a deterministic master manifest
#   6. Materialize processed train/val/test folders using
#      hardlinks by default to avoid duplicating gigabytes of data
#   7. Save split and preparation reports
#
# This script does NOT modify raw MinneApple data.
# ============================================================


SPLIT_ORDER = {
    "train": 0,
    "val": 1,
    "test": 2,
}


MANIFEST_FIELDS = [
    "sample_id",
    "split",
    "source_split",
    "group_id",
    "image_stem",
    "image_name",
    "mask_name",
    "image_path",
    "mask_path",
    "processed_image_path",
    "processed_mask_path",
    "image_width",
    "image_height",
    "instance_count",
]


# ============================================================
# Helpers
# ============================================================


def relative_path(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def read_csv(path):
    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    path = Path(path)
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
        writer.writerows(rows)


def parse_int(value, field_name, row_name):
    try:
        return int(value)
    except Exception as error:
        raise ValueError(
            f"Invalid integer for {field_name} in {row_name}: {value}"
        ) from error


def normalize_audit_relative_path(project_root, value):
    path = Path(value)

    if path.is_absolute():
        return path

    return (project_root / path).resolve()


def infer_group_id(stem):
    # MinneApple training filenames commonly follow patterns such as:
    #   20150921_131346_image736
    # Images sharing the acquisition prefix are kept in the same split.
    lower_stem = stem.lower()
    marker = "_image"
    position = lower_stem.rfind(marker)

    if position > 0:
        suffix = lower_stem[position + len(marker):]

        if suffix.isdigit():
            group = stem[:position]
            return group

    # Conservative generic fallback:
    # remove a trailing purely numeric token separated by underscore/hyphen.
    for separator in ("_", "-"):
        if separator in stem:
            prefix, suffix = stem.rsplit(separator, 1)

            if prefix and suffix.isdigit():
                return prefix

    # Safest final fallback: treat the image as its own group.
    return stem


# ============================================================
# Audit loading and validation
# ============================================================


def validate_audit_summary(audit_summary):
    status = str(audit_summary.get("status", "")).upper()

    if status not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise RuntimeError(
            "Dataset audit has not passed. "
            f"Current audit status: {status or 'UNKNOWN'}"
        )

    if audit_summary.get("errors"):
        raise RuntimeError(
            "Audit summary still contains errors. "
            "Resolve them before preparing the dataset."
        )

    datasets = audit_summary.get("datasets", [])

    if not datasets:
        raise RuntimeError(
            "Audit summary contains no discovered datasets."
        )

    train_roots = [
        item
        for item in datasets
        if item.get("split_hint") == "train"
    ]

    test_roots = [
        item
        for item in datasets
        if item.get("split_hint") == "test"
    ]

    if len(train_roots) != 1:
        raise RuntimeError(
            "Expected exactly one audited official training dataset root, "
            f"but found {len(train_roots)}."
        )

    if len(test_roots) != 1:
        raise RuntimeError(
            "Expected exactly one audited official test dataset root, "
            f"but found {len(test_roots)}."
        )

    return train_roots[0], test_roots[0]


def load_audited_samples(project_root, audit_csv_path):
    rows = read_csv(audit_csv_path)

    samples = []

    for row in rows:
        status = str(row.get("status", "")).upper()

        if status != "PASS":
            raise RuntimeError(
                "Audit CSV contains a non-PASS image/mask pair: "
                f"{row.get('stem', '<unknown>')} -> {status}"
            )

        split_hint = str(row.get("split_hint", "")).lower()

        if split_hint not in {"train", "test"}:
            raise RuntimeError(
                "Unexpected split_hint in audit CSV: "
                f"{split_hint} for {row.get('stem', '<unknown>')}"
            )

        image_path = normalize_audit_relative_path(
            project_root,
            row["image_path"],
        )

        mask_path = normalize_audit_relative_path(
            project_root,
            row["mask_path"],
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Audited image is missing: {image_path}"
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Audited mask is missing: {mask_path}"
            )

        stem = row["stem"]

        sample = {
            "source_split": (
                "official_train"
                if split_hint == "train"
                else "official_test"
            ),
            "split_hint": split_hint,
            "group_id": (
                infer_group_id(stem)
                if split_hint == "train"
                else "official_test"
            ),
            "image_stem": stem,
            "image_path_abs": image_path,
            "mask_path_abs": mask_path,
            "image_name": image_path.name,
            "mask_name": mask_path.name,
            "image_width": parse_int(
                row["image_width"],
                "image_width",
                stem,
            ),
            "image_height": parse_int(
                row["image_height"],
                "image_height",
                stem,
            ),
            "instance_count": parse_int(
                row["instance_count"],
                "instance_count",
                stem,
            ),
        }

        samples.append(sample)

    return samples


# ============================================================
# Group-aware train/validation split
# ============================================================


def build_training_groups(train_samples):
    groups = defaultdict(list)

    for sample in train_samples:
        groups[sample["group_id"]].append(sample)

    return dict(groups)


def choose_validation_groups(groups, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            f"val_ratio must be between 0 and 1, got {val_ratio}"
        )

    if len(groups) < 2:
        raise RuntimeError(
            "Fewer than two acquisition groups were discovered in the "
            "official training set. A group-aware validation split cannot "
            "be created safely."
        )

    total_images = sum(len(samples) for samples in groups.values())
    target_images = int(round(total_images * val_ratio))

    items = [
        (group_id, len(samples))
        for group_id, samples in groups.items()
    ]

    rng = random.Random(seed)
    rng.shuffle(items)

    # Dynamic programming over image counts.
    # dp[count] stores one deterministic tuple of selected group IDs that
    # reaches exactly 'count' validation images.
    dp = {0: tuple()}

    for group_id, group_size in items:
        snapshot = list(dp.items())

        for current_count, selected_groups in reversed(snapshot):
            new_count = current_count + group_size

            if new_count > total_images:
                continue

            if new_count not in dp:
                dp[new_count] = selected_groups + (group_id,)

    candidate_counts = [
        count
        for count in dp.keys()
        if 0 < count < total_images
    ]

    if not candidate_counts:
        raise RuntimeError(
            "Could not construct a non-empty group-aware validation split."
        )

    best_count = min(
        candidate_counts,
        key=lambda count: (
            abs(count - target_images),
            count > target_images,
            count,
        ),
    )

    selected = set(dp[best_count])

    return selected, target_images, best_count


# ============================================================
# Processed dataset materialization
# ============================================================


def ensure_empty_or_compatible_destination(source, destination, force):
    if not destination.exists():
        return "new"

    if force:
        destination.unlink()
        return "replace"

    try:
        if os.path.samefile(source, destination):
            return "existing_samefile"
    except OSError:
        pass

    if destination.stat().st_size == source.stat().st_size:
        return "existing_same_size"

    raise FileExistsError(
        "Destination already exists and does not match the source. "
        f"Use --force to rebuild processed files:\n{destination}"
    )


def materialize_file(source, destination, mode, force=False):
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing_state = ensure_empty_or_compatible_destination(
        source,
        destination,
        force,
    )

    if existing_state in {"existing_samefile", "existing_same_size"}:
        return "existing"

    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"

    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"

    if mode == "none":
        return "none"

    raise ValueError(f"Unsupported materialize mode: {mode}")


def check_destination_collisions(samples, processed_root):
    seen = {}

    for sample in samples:
        split = sample["split"]

        for kind, name in (
            ("images", sample["image_name"]),
            ("masks", sample["mask_name"]),
        ):
            destination = processed_root / split / kind / name
            key = str(destination).lower()

            if key in seen:
                previous = seen[key]
                raise RuntimeError(
                    "Destination filename collision detected:\n"
                    f"  {previous}\n"
                    f"  {sample['image_stem']}\n"
                    f"Destination: {destination}"
                )

            seen[key] = sample["image_stem"]


def materialize_dataset(samples, processed_root, project_root, mode, force):
    stats = defaultdict(int)

    for index, sample in enumerate(samples, start=1):
        split = sample["split"]

        image_destination = (
            processed_root
            / split
            / "images"
            / sample["image_name"]
        )

        mask_destination = (
            processed_root
            / split
            / "masks"
            / sample["mask_name"]
        )

        if mode != "none":
            image_result = materialize_file(
                sample["image_path_abs"],
                image_destination,
                mode,
                force=force,
            )

            mask_result = materialize_file(
                sample["mask_path_abs"],
                mask_destination,
                mode,
                force=force,
            )

            stats[f"image_{image_result}"] += 1
            stats[f"mask_{mask_result}"] += 1

        sample["processed_image_path"] = (
            relative_path(image_destination, project_root)
            if mode != "none"
            else ""
        )

        sample["processed_mask_path"] = (
            relative_path(mask_destination, project_root)
            if mode != "none"
            else ""
        )

        if index % 100 == 0 or index == len(samples):
            print(
                f"      Prepared {index}/{len(samples)} samples"
            )

    return dict(stats)


# ============================================================
# Manifest and summary creation
# ============================================================


def assign_final_splits(samples, validation_groups):
    for sample in samples:
        if sample["source_split"] == "official_test":
            sample["split"] = "test"
        else:
            sample["split"] = (
                "val"
                if sample["group_id"] in validation_groups
                else "train"
            )


def sort_samples(samples):
    return sorted(
        samples,
        key=lambda item: (
            SPLIT_ORDER[item["split"]],
            item["group_id"].lower(),
            item["image_name"].lower(),
        ),
    )


def assign_sample_ids(samples):
    counters = defaultdict(int)

    for sample in samples:
        split = sample["split"]
        counters[split] += 1
        sample["sample_id"] = (
            f"{split}_{counters[split]:06d}"
        )


def build_manifest_rows(samples, project_root):
    rows = []

    for sample in samples:
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "source_split": sample["source_split"],
                "group_id": sample["group_id"],
                "image_stem": sample["image_stem"],
                "image_name": sample["image_name"],
                "mask_name": sample["mask_name"],
                "image_path": relative_path(
                    sample["image_path_abs"],
                    project_root,
                ),
                "mask_path": relative_path(
                    sample["mask_path_abs"],
                    project_root,
                ),
                "processed_image_path": sample.get(
                    "processed_image_path",
                    "",
                ),
                "processed_mask_path": sample.get(
                    "processed_mask_path",
                    "",
                ),
                "image_width": sample["image_width"],
                "image_height": sample["image_height"],
                "instance_count": sample["instance_count"],
            }
        )

    return rows


def build_group_rows(groups, validation_groups):
    rows = []

    for group_id in sorted(groups.keys(), key=str.lower):
        samples = groups[group_id]

        rows.append(
            {
                "group_id": group_id,
                "num_images": len(samples),
                "num_instances": sum(
                    sample["instance_count"]
                    for sample in samples
                ),
                "assigned_split": (
                    "val"
                    if group_id in validation_groups
                    else "train"
                ),
                "example_image": samples[0]["image_name"],
            }
        )

    return rows


def summarize_split(samples, split):
    selected = [
        sample
        for sample in samples
        if sample["split"] == split
    ]

    instances = sum(
        sample["instance_count"]
        for sample in selected
    )

    return {
        "images": len(selected),
        "instances": instances,
        "mean_instances_per_image": (
            round(instances / len(selected), 3)
            if selected
            else 0.0
        ),
        "groups": len(
            {
                sample["group_id"]
                for sample in selected
                if sample["source_split"] == "official_train"
            }
        ),
    }


def verify_final_split(samples, expected_train_total, expected_test_total):
    official_train = [
        sample
        for sample in samples
        if sample["source_split"] == "official_train"
    ]

    official_test = [
        sample
        for sample in samples
        if sample["source_split"] == "official_test"
    ]

    if len(official_train) != expected_train_total:
        raise RuntimeError(
            "Official training sample count differs from audit summary: "
            f"{len(official_train)} vs {expected_train_total}"
        )

    if len(official_test) != expected_test_total:
        raise RuntimeError(
            "Official test sample count differs from audit summary: "
            f"{len(official_test)} vs {expected_test_total}"
        )

    if any(sample["split"] != "test" for sample in official_test):
        raise RuntimeError(
            "Official test leakage detected: one or more official test "
            "samples were assigned outside the final test split."
        )

    train_groups = {
        sample["group_id"]
        for sample in samples
        if sample["split"] == "train"
        and sample["source_split"] == "official_train"
    }

    val_groups = {
        sample["group_id"]
        for sample in samples
        if sample["split"] == "val"
        and sample["source_split"] == "official_train"
    }

    overlap = train_groups & val_groups

    if overlap:
        raise RuntimeError(
            "Group leakage detected between train and validation: "
            f"{sorted(overlap)}"
        )

    stems = [sample["image_stem"] for sample in samples]

    if len(stems) != len(set(stems)):
        raise RuntimeError(
            "Duplicate image stems found across the prepared dataset."
        )


# ============================================================
# Metadata compatibility
# ============================================================


def preparation_signature(val_ratio, seed, materialize_mode):
    return {
        "val_ratio": round(float(val_ratio), 8),
        "seed": int(seed),
        "materialize_mode": materialize_mode,
        "grouping": "acquisition_prefix_auto",
    }


def check_previous_preparation(metadata_path, signature, force):
    if not metadata_path.exists():
        return

    previous = read_json(metadata_path)
    previous_signature = previous.get("signature", {})

    if previous_signature == signature:
        return

    if force:
        return

    raise RuntimeError(
        "The processed dataset was previously prepared with different "
        "settings. Use --force to rebuild it.\n"
        f"Previous settings: {previous_signature}\n"
        f"Requested settings: {signature}"
    )


def clear_processed_split_folders(processed_root):
    for split in ("train", "val", "test"):
        split_root = processed_root / split

        if split_root.exists():
            shutil.rmtree(split_root)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the audited MinneApple dataset for SAM 3 experiments."
        )
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Optional project root override. By default the script assumes "
            "it lives in AgriAnnotationSystem/scripts/."
        ),
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help=(
            "Target validation fraction of the official training set. "
            "The actual fraction may differ slightly because acquisition "
            "groups are never split. Default: 0.20"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used only for deterministic tie-breaking.",
    )

    parser.add_argument(
        "--materialize-mode",
        choices=["hardlink", "copy", "none"],
        default="copy",
        help=(
            "How to populate data/processed/minneapple. 'copy' creates an "
            "independent processed dataset and is the safest default. "
            "'hardlink' saves disk space but must be treated as read-only. "
            "Default: copy"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild processed train/val/test folders even if a previous "
            "preparation exists. Raw data are never deleted."
        ),
    )

    args = parser.parse_args()

    if args.project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(args.project_root).resolve()

    audit_root = (
        project_root
        / "outputs"
        / "reports"
        / "minneapple_audit"
    )

    audit_summary_path = audit_root / "audit_summary.json"
    audit_csv_path = audit_root / "image_mask_audit.csv"

    split_root = project_root / "data" / "splits"
    processed_root = (
        project_root
        / "data"
        / "processed"
        / "minneapple"
    )

    report_root = (
        project_root
        / "outputs"
        / "reports"
        / "minneapple_prepare"
    )

    manifest_path = split_root / "minneapple_split.csv"
    group_path = split_root / "minneapple_train_groups.csv"
    summary_path = report_root / "prepare_summary.json"
    metadata_path = processed_root / ".prepare_metadata.json"

    print()
    print("=" * 78)
    print("MINNEAPPLE DATASET PREPARATION")
    print("=" * 78)
    print(f"Project root       : {project_root}")
    print(f"Audit summary      : {audit_summary_path}")
    print(f"Audit pair CSV     : {audit_csv_path}")
    print(f"Processed root     : {processed_root}")
    print(f"Validation target  : {args.val_ratio:.2%}")
    print(f"Seed               : {args.seed}")
    print(f"Materialize mode   : {args.materialize_mode}")
    print()

    if not audit_summary_path.is_file():
        raise FileNotFoundError(
            "Missing audit summary. Run scripts/01_audit_minneapple.py first:\n"
            f"{audit_summary_path}"
        )

    if not audit_csv_path.is_file():
        raise FileNotFoundError(
            "Missing image/mask audit CSV. Run 01_audit_minneapple.py first:\n"
            f"{audit_csv_path}"
        )

    # --------------------------------------------------------
    # Stage 1: Validate audit result
    # --------------------------------------------------------

    print("[1/7] Validating Stage 1 audit...")

    audit_summary = read_json(audit_summary_path)
    train_audit, test_audit = validate_audit_summary(audit_summary)

    expected_train_total = int(train_audit["paired_count"])
    expected_test_total = int(test_audit["paired_count"])

    print(f"  [OK] Audit status       : {audit_summary['status']}")
    print(f"  [OK] Official train     : {expected_train_total}")
    print(f"  [OK] Official test      : {expected_test_total}")
    print(f"  [OK] Total audited      : {expected_train_total + expected_test_total}")

    # --------------------------------------------------------
    # Stage 2: Load audited pairs
    # --------------------------------------------------------

    print()
    print("[2/7] Loading audited image/mask pairs...")

    samples = load_audited_samples(
        project_root,
        audit_csv_path,
    )

    train_samples = [
        sample
        for sample in samples
        if sample["source_split"] == "official_train"
    ]

    test_samples = [
        sample
        for sample in samples
        if sample["source_split"] == "official_test"
    ]

    if len(train_samples) != expected_train_total:
        raise RuntimeError(
            "Audit CSV train count does not match audit summary: "
            f"{len(train_samples)} vs {expected_train_total}"
        )

    if len(test_samples) != expected_test_total:
        raise RuntimeError(
            "Audit CSV test count does not match audit summary: "
            f"{len(test_samples)} vs {expected_test_total}"
        )

    print(f"  [OK] Loaded train pairs : {len(train_samples)}")
    print(f"  [OK] Loaded test pairs  : {len(test_samples)}")

    # --------------------------------------------------------
    # Stage 3: Group-aware train/validation split
    # --------------------------------------------------------

    print()
    print("[3/7] Building group-aware train/validation split...")

    groups = build_training_groups(train_samples)

    validation_groups, target_val_images, actual_val_images = (
        choose_validation_groups(
            groups,
            args.val_ratio,
            args.seed,
        )
    )

    assign_final_splits(
        samples,
        validation_groups,
    )

    samples = sort_samples(samples)
    assign_sample_ids(samples)

    verify_final_split(
        samples,
        expected_train_total,
        expected_test_total,
    )

    train_summary = summarize_split(samples, "train")
    val_summary = summarize_split(samples, "val")
    test_summary = summarize_split(samples, "test")

    print(f"  Acquisition groups      : {len(groups)}")
    print(f"  Validation groups       : {len(validation_groups)}")
    print(f"  Target val images       : {target_val_images}")
    print(f"  Actual val images       : {actual_val_images}")
    print(f"  Final train images      : {train_summary['images']}")
    print(f"  Final val images        : {val_summary['images']}")
    print(f"  Official test images    : {test_summary['images']}")
    print("  [OK] No train/val group overlap")
    print("  [OK] Official test set remains sealed")

    # --------------------------------------------------------
    # Stage 4: Check previous preparation
    # --------------------------------------------------------

    print()
    print("[4/7] Checking preparation metadata...")

    signature = preparation_signature(
        args.val_ratio,
        args.seed,
        args.materialize_mode,
    )

    check_previous_preparation(
        metadata_path,
        signature,
        args.force,
    )

    if args.force and args.materialize_mode != "none":
        print("  [INFO] --force enabled; rebuilding processed folders")
        clear_processed_split_folders(processed_root)
    else:
        print("  [OK] Preparation settings are compatible")

    # --------------------------------------------------------
    # Stage 5: Materialize processed dataset
    # --------------------------------------------------------

    print()
    print("[5/7] Preparing processed train/val/test folders...")

    check_destination_collisions(
        samples,
        processed_root,
    )

    materialization_stats = materialize_dataset(
        samples,
        processed_root,
        project_root,
        args.materialize_mode,
        force=args.force,
    )

    if args.materialize_mode == "none":
        print("  [SKIP] Files were not materialized (--materialize-mode none)")
    else:
        print("  [OK] Processed dataset prepared")

    # --------------------------------------------------------
    # Stage 6: Write manifests
    # --------------------------------------------------------

    print()
    print("[6/7] Writing split manifests...")

    manifest_rows = build_manifest_rows(
        samples,
        project_root,
    )

    group_rows = build_group_rows(
        groups,
        validation_groups,
    )

    write_csv(
        manifest_path,
        manifest_rows,
        MANIFEST_FIELDS,
    )

    write_csv(
        group_path,
        group_rows,
        [
            "group_id",
            "num_images",
            "num_instances",
            "assigned_split",
            "example_image",
        ],
    )

    print(f"  [OK] Main manifest      : {manifest_path}")
    print(f"  [OK] Group manifest     : {group_path}")

    # --------------------------------------------------------
    # Stage 7: Save preparation summary
    # --------------------------------------------------------

    print()
    print("[7/7] Saving preparation summary...")

    summary = {
        "prepare_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "status": "PASS",
        "project_root": str(project_root),
        "audit_summary": relative_path(audit_summary_path, project_root),
        "audit_pair_csv": relative_path(audit_csv_path, project_root),
        "signature": signature,
        "official_source": {
            "train_images": expected_train_total,
            "test_images": expected_test_total,
            "official_test_preserved": True,
        },
        "grouping": {
            "method": "acquisition_prefix_auto",
            "num_training_groups": len(groups),
            "num_validation_groups": len(validation_groups),
            "target_validation_images": target_val_images,
            "actual_validation_images": actual_val_images,
            "train_val_group_overlap": 0,
        },
        "splits": {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        },
        "totals": {
            "images": len(samples),
            "instances": sum(
                sample["instance_count"]
                for sample in samples
            ),
        },
        "materialization": {
            "mode": args.materialize_mode,
            "processed_root": relative_path(processed_root, project_root),
            "stats": materialization_stats,
        },
        "outputs": {
            "manifest": relative_path(manifest_path, project_root),
            "group_manifest": relative_path(group_path, project_root),
            "summary": relative_path(summary_path, project_root),
        },
    }

    write_json(summary_path, summary)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "signature": signature,
        "manifest": relative_path(manifest_path, project_root),
        "summary": relative_path(summary_path, project_root),
    }

    write_json(metadata_path, metadata)

    print(f"  [OK] Summary            : {summary_path}")
    print(f"  [OK] Metadata           : {metadata_path}")

    print()
    print("=" * 78)
    print("MINNEAPPLE PREPARATION COMPLETE")
    print("=" * 78)
    print("STATUS                  : PASS")
    print(f"TRAIN IMAGES            : {train_summary['images']}")
    print(f"TRAIN APPLE INSTANCES   : {train_summary['instances']}")
    print(f"VAL IMAGES              : {val_summary['images']}")
    print(f"VAL APPLE INSTANCES     : {val_summary['instances']}")
    print(f"TEST IMAGES             : {test_summary['images']}")
    print(f"TEST APPLE INSTANCES    : {test_summary['instances']}")
    print(f"TOTAL IMAGES            : {len(samples)}")
    print(
        "TOTAL APPLE INSTANCES   : "
        f"{sum(sample['instance_count'] for sample in samples)}"
    )
    print(f"TRAIN GROUPS            : {train_summary['groups']}")
    print(f"VAL GROUPS              : {val_summary['groups']}")
    print("TRAIN/VAL GROUP OVERLAP : 0")
    print("OFFICIAL TEST PRESERVED : YES")
    print(f"MATERIALIZE MODE        : {args.materialize_mode}")
    print(f"MANIFEST                : {manifest_path}")
    print(f"SUMMARY                 : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user.")
        sys.exit(130)
    except Exception as error:
        print()
        print("=" * 78)
        print("MINNEAPPLE PREPARATION FAILED")
        print("=" * 78)
        print(f"ERROR: {error}")
        print("=" * 78)
        raise
