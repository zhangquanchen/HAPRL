#!/usr/bin/env python3
"""
Merge per-sample annotation files exported by the labelling tool into a
single qa_process training file. Raw per-sample exports are not bundled
in this package; the merged files already live in data/mydata/train_final/.

Input format (one JSON object per file):
  - id, image, instruction, type, solution, final_bboxes, events

Output format (qa_process list):
  - images, doc_id, data_source, extra_info, problem, solution

Usage:
  python scripts/rl/merge_qa_process_data.py --input_dirs raw_exports
  python scripts/rl/merge_qa_process_data.py --input_dirs raw_exports --output_name merged_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Defaults are resolved against the package root (the parent of scripts/).
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = "train_final"
DEFAULT_OUTPUT_NAME = "train.json"


def load_single_sample(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load one annotation file; return None if it cannot be parsed."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] skipping {filepath}: {e}")
        return None


def convert_to_qa_process(raw: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    """Convert one raw annotation record into the qa_process sample schema."""
    image = raw.get("image", "")
    images = [image] if image else []

    problem = raw.get("instruction", "")
    solution = raw.get("solution", "")

    extra_info: Dict[str, Any] = {"events": raw.get("events", [])}

    if "annotation_time" in raw:
        extra_info["annotation_time"] = raw["annotation_time"]
    else:
        extra_info["annotation_time"] = 0.0

    if "task_evaluation" in raw:
        extra_info["task_evaluation"] = raw["task_evaluation"]
    else:
        extra_info["task_evaluation"] = {
            "difficulty_score": 0.0,
            "difficulty_level": "unknown",
            "is_ambiguous": False,
            "confidence_score": 0.0,
        }

    # Final evidence boxes are consumed by the grounding reward and by the
    # process judge as a reference region.
    if "final_bboxes" in raw:
        extra_info["final_bboxes"] = raw["final_bboxes"]

    return {
        "images": images,
        "doc_id": doc_id,
        "data_source": "qa_process",
        "extra_info": extra_info,
        "problem": problem,
        "solution": solution,
    }


def merge_data(
    input_dirs: List[str],
    mydata_dir: Path,
    output_dir: str,
    output_name: str,
) -> Path:
    """Merge every annotation file under the given directories into one file."""
    merged: List[Dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    for input_dir in input_dirs:
        src = mydata_dir / input_dir
        if not src.exists():
            print(f"[WARN] directory does not exist, skipping: {src}")
            continue

        json_files = sorted(src.glob("*.json"))
        print(f"[INFO] {input_dir}: found {len(json_files)} annotation files")

        for fp in json_files:
            raw = load_single_sample(fp)
            if raw is None:
                continue

            # The file stem is used as doc_id; collisions get a numeric suffix.
            doc_id = fp.stem
            if doc_id in seen_doc_ids:
                i = 1
                while f"{doc_id}_{i}" in seen_doc_ids:
                    i += 1
                doc_id = f"{doc_id}_{i}"
            seen_doc_ids.add(doc_id)

            merged.append(convert_to_qa_process(raw, doc_id))

    out_dir = mydata_dir / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[INFO] merged {len(merged)} samples")
    print(f"[INFO] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-sample annotation files into a qa_process dataset"
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="directory names under mydata that contain one-object-per-file JSON exports",
    )
    parser.add_argument(
        "--mydata",
        type=str,
        default=None,
        help="path to the mydata directory, default: <package root>/data/mydata",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory name under mydata, default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help=f"output file name, default: {DEFAULT_OUTPUT_NAME}",
    )
    args = parser.parse_args()

    mydata_dir = Path(args.mydata) if args.mydata else (ROOT / "data" / "mydata")
    merge_data(
        input_dirs=args.input_dirs,
        mydata_dir=mydata_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
