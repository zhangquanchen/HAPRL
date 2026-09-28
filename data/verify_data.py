#!/usr/bin/env python3
"""Check the bundled annotation files before starting a training or evaluation run.

Validates JSON syntax, the field schema of every split, uniqueness of doc_id,
and the fact that all image references are relative paths. Image files are not
shipped with this package, so existence checks are opt-in via --check-images.

Usage:
    python3 data/verify_data.py
    python3 data/verify_data.py --check-images
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

DATA_ROOT = Path(__file__).resolve().parent / "mydata"

# Splits that follow the flat list-of-samples schema.
LIST_SPLITS = (
    "qa_data/train.json",
    "qa_data/val.json",
    "qa_data/val_mini.json",
    "qa_data/test.json",
    "qa_process/train.json",
    "qa_process/val.json",
    "qa_process/val_mini.json",
    "qa_process/val_mini_acc.json",
    "qa_process/test.json",
    "iou_data/train.json",
    "iou_data/val.json",
    "iou_data/test.json",
    "iou_process/train.json",
    "iou_process/val.json",
    "iou_process/test.json",
    "train_final/train.json",
    "train_final/train1000.json",
    "train_final/train1000_v1.json",
)

SAMPLE_KEYS = {"images", "doc_id", "data_source", "problem", "solution", "extra_info"}

MACHINE_PATH_PATTERNS = {
    "windows drive path": re.compile(r"[A-Za-z]:[\\/]{1,2}(?:Users|Documents|Desktop)", re.I),
    "macos home path": re.compile(r"/Users/[^/\s\"]+/"),
    "linux home path": re.compile(r"/home/[^/\s\"]+/"),
    "cluster user path": re.compile(r"/data/(?:user|users|home)/[^/\s\"]+/", re.I),
    "internal mount path": re.compile(r"/mnt/[^\s\"]+", re.I),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "private ip address": re.compile(
        r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
    "credential literal": re.compile(r"\b(?:sk|hf|ghp|gho)_[A-Za-z0-9]{16,}"),
}


def iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def is_relative_image_path(value: str) -> bool:
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def scan_machine_paths(value: Any, label: str) -> list[str]:
    for text in iter_strings(value):
        for name, pattern in MACHINE_PATH_PATTERNS.items():
            if pattern.search(text):
                return [f"{label}: {name} found in free text"]
    return []


def check_sample(
    sample: Any,
    label: str,
    seen_doc_ids: set[str],
    images: set[str],
) -> list[str]:
    if not isinstance(sample, dict):
        return [f"{label}: sample must be a JSON object"]

    errors: list[str] = []
    missing = SAMPLE_KEYS - sample.keys()
    if missing:
        errors.append(f"{label}: missing fields {sorted(missing)}")

    doc_id = sample.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        errors.append(f"{label}: doc_id must be a non-empty string")
    else:
        key = f"{label.split('[')[0]}::{doc_id}"
        if key in seen_doc_ids:
            errors.append(f"{label}: duplicate doc_id {doc_id!r} within the split")
        seen_doc_ids.add(key)

    for field in ("data_source", "problem", "solution"):
        if not isinstance(sample.get(field), str) or not sample[field]:
            errors.append(f"{label}: {field} must be a non-empty string")

    image_list = sample.get("images")
    if not isinstance(image_list, list) or not image_list:
        errors.append(f"{label}: images must be a non-empty list")
    else:
        for image in image_list:
            if not isinstance(image, str) or not is_relative_image_path(image):
                errors.append(f"{label}: image reference must be a relative path")
            else:
                images.add(image)

    if not isinstance(sample.get("extra_info"), dict):
        errors.append(f"{label}: extra_info must be a JSON object")

    errors.extend(scan_machine_paths(sample, label))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="also require every referenced image file to exist under data/mydata",
    )
    args = parser.parse_args()

    errors: list[str] = []
    counts: dict[str, int] = {}
    images: set[str] = set()
    seen_doc_ids: set[str] = set()

    for relative in LIST_SPLITS:
        path = DATA_ROOT / relative
        if not path.is_file():
            errors.append(f"{relative}: file is missing")
            continue
        try:
            samples = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relative}: not valid UTF-8 JSON ({exc})")
            continue
        if not isinstance(samples, list):
            errors.append(f"{relative}: the top level must be a JSON list")
            continue
        counts[relative] = len(samples)
        for index, sample in enumerate(samples):
            errors.extend(
                check_sample(sample, f"{relative}[{index}]", seen_doc_ids, images)
            )

    if args.check_images:
        missing_images = sorted(
            image for image in images if not (DATA_ROOT / image).is_file()
        )
        if missing_images:
            errors.append(
                f"{len(missing_images)} referenced image files are missing, "
                f"first: {missing_images[0]} (see the top-level README for the layout)"
            )

    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more", file=sys.stderr)
        return 1

    for relative, count in counts.items():
        print(f"  {relative:<34} {count:>5} samples")
    print(f"\nOK: {len(images)} unique image references, all relative.")
    if not args.check_images:
        print("Image files were not checked. Re-run with --check-images once you "
              "have populated data/mydata/data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
