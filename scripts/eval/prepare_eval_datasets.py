#!/usr/bin/env python3
"""Prepare external benchmarks for Mini-o3/verl val-only evaluation.

The output schema is the same JSON-list format used by the project:
images/problem/solution/doc_id/data_source. QA datasets use data_source=qa_data
so evaluation only calls the answer-accuracy judge. VStar bbox evaluation uses
data_source=iou_data and is scored locally by iou_bbox.py.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any


CHOICE_KEYS = ("A", "B", "C", "D", "E")
csv.field_size_limit(sys.maxsize)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def rel(path: Path, data_root: Path) -> str:
    return path.resolve().relative_to(data_root.resolve()).as_posix()


def add_image_token(text: str) -> str:
    text = (text or "").strip()
    return text if text.startswith("<image>") else f"<image>\n{text}"


def normalize_visualprobe(src: Path, data_root: Path, data_source_name: str) -> list[dict[str, Any]]:
    rows = []
    for i, row in enumerate(read_json(src)):
        images = row.get("images") or []
        rows.append(
            {
                "images": images,
                "doc_id": row.get("doc_id") or f"{data_source_name}_{i}",
                "problem": add_image_token(row.get("problem", "")),
                "solution": row.get("solution", ""),
                "data_source": "qa_data",
                "extra_info": {
                    "source_data_source": row.get("data_source", data_source_name),
                    "source_path": rel(src, data_root),
                },
            }
        )
    return rows


def normalize_vstar_qa(src: Path, data_root: Path) -> list[dict[str, Any]]:
    rows = []
    root = src.parent
    with src.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            image_rel = f"Vstar_Bench/{row['image']}"
            rows.append(
                {
                    "images": [image_rel],
                    "doc_id": f"vstar_{row.get('category', 'unknown')}_{row.get('question_id', line_idx)}",
                    "problem": add_image_token(row.get("text", "")),
                    "solution": row.get("label", ""),
                    "data_source": "qa_data",
                    "extra_info": {
                        "source_data_source": "vstar_qa",
                        "category": row.get("category", ""),
                        "question_id": row.get("question_id", ""),
                        "source_path": rel(src, data_root),
                    },
                }
            )
    return rows


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def find_vstar_bbox_rows(src: Path, data_root: Path) -> list[dict[str, Any]]:
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            image_rel = f"Vstar_Bench/{row['image']}"
            image_path = data_root / image_rel
            meta_path = image_path.with_suffix(".json")
            if not meta_path.exists():
                continue
            meta = read_json(meta_path)
            bboxes = meta.get("bbox") or []
            if not bboxes:
                continue
            size = image_size(image_path)
            if size is None:
                continue
            width, height = size
            x, y, w, h = [float(v) for v in bboxes[0]]
            x1 = max(0.0, min(1.0, x / width))
            y1 = max(0.0, min(1.0, y / height))
            x2 = max(0.0, min(1.0, (x + w) / width))
            y2 = max(0.0, min(1.0, (y + h) / height))
            target = ", ".join(meta.get("target_object") or ["target object"])
            problem = (
                "<image>\n"
                f"Locate the {target} relevant to the question: {meta.get('question', row.get('text', ''))}\n"
                "Return only the normalized bounding box of the target object in this exact format: "
                "<answer><bbox>x1,y1,x2,y2</bbox></answer>. Coordinates must be relative numbers in [0,1]."
            )
            rows.append(
                {
                    "images": [image_rel],
                    "doc_id": f"vstar_bbox_{row.get('category', 'unknown')}_{row.get('question_id', line_idx)}",
                    "problem": problem,
                    "solution": f"<bbox>{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}</bbox>",
                    "data_source": "iou_data",
                    "extra_info": {
                        "source_data_source": "vstar_bbox",
                        "category": row.get("category", ""),
                        "question_id": row.get("question_id", ""),
                        "target_object": meta.get("target_object", []),
                        "pixel_bbox_xywh": bboxes[0],
                        "image_size": [width, height],
                        "source_path": rel(src, data_root),
                    },
                }
            )
    return rows


def decode_hr_image(image_text: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    try:
        raw = base64.b64decode(image_text)
    except Exception:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return True


def normalize_hrbench(tsv_path: Path, data_root: Path, out_image_dir: Path, split_name: str) -> list[dict[str, Any]]:
    rows = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = row.get("index", str(len(rows)))
            image_out = out_image_dir / split_name / f"{idx}.jpg"
            if not decode_hr_image(row.get("image", ""), image_out):
                continue
            choices = "\n".join(f"({key}) {row[key]}" for key in ("A", "B", "C", "D") if row.get(key))
            question = f"{row.get('question', '').strip()}\n{choices}\nAnswer with the option's letter directly."
            rows.append(
                {
                    "images": [rel(image_out, data_root)],
                    "doc_id": f"hrbench_{split_name}_{idx}",
                    "problem": add_image_token(question),
                    "solution": row.get("answer", ""),
                    "data_source": "qa_data",
                    "extra_info": {
                        "source_data_source": f"hrbench_{split_name}",
                        "category": row.get("category", ""),
                        "cycle_category": row.get("cycle_category", ""),
                        "source_path": rel(tsv_path, data_root),
                    },
                }
            )
    return rows


def normalize_mme(src: Path, data_root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    missing_images = 0
    for i, row in enumerate(read_json(src)):
        choices = row.get("Answer choices") or []
        choice_text = "\n".join(str(x) for x in choices)
        problem = (
            f"{row.get('Text', '').strip()}\n{choice_text}\n"
            "Answer with the option's letter directly."
        )
        image_rel = f"MME-RealWorld/{row.get('Image', '')}"
        if not (data_root / image_rel).exists():
            missing_images += 1
            continue
        rows.append(
            {
                "images": [image_rel],
                "doc_id": f"mme_realworld_{row.get('Question_id', i)}".replace("/", "_"),
                "problem": add_image_token(problem),
                "solution": row.get("Ground truth", ""),
                "data_source": "qa_data",
                "extra_info": {
                    "source_data_source": "mme_realworld",
                    "task": row.get("Task", ""),
                    "subtask": row.get("Subtask", ""),
                    "category": row.get("Category", ""),
                    "dataset": row.get("Dataset", ""),
                    "source_path": rel(src, data_root),
                },
            }
        )
    return rows, missing_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Root containing downloaded benchmark data")
    parser.add_argument("--out-dir", required=True, help="Directory for normalized evaluation JSON")
    parser.add_argument("--hr-split", choices=["4k", "8k", "both"], default="4k")
    parser.add_argument("--mme-max-samples", type=int, default=0, help="0 means keep all samples")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}

    visual_sets = {
        "visualprobe_easy": data_root / "VisualProbe_Easy" / "val.json",
        "visualprobe_medium": data_root / "VisualProbe_Medium" / "val.json",
        "visualprobe_hard": data_root / "VisualProbe_Hard" / "val.json",
    }
    for name, src in visual_sets.items():
        rows = normalize_visualprobe(src, data_root, name)
        write_json(out_dir / f"{name}.json", rows)
        summary[name] = {"rows": len(rows), "path": str(out_dir / f"{name}.json")}

    vstar_src = data_root / "Vstar_Bench" / "test_questions.jsonl"
    vstar_qa = normalize_vstar_qa(vstar_src, data_root)
    write_json(out_dir / "vstar_qa.json", vstar_qa)
    summary["vstar_qa"] = {"rows": len(vstar_qa), "path": str(out_dir / "vstar_qa.json")}
    vstar_bbox = find_vstar_bbox_rows(vstar_src, data_root)
    write_json(out_dir / "vstar_bbox.json", vstar_bbox)
    summary["vstar_bbox"] = {"rows": len(vstar_bbox), "path": str(out_dir / "vstar_bbox.json")}

    hr_image_dir = data_root / "HR-Bench" / "decoded_images"
    if args.hr_split in ("4k", "both"):
        rows = normalize_hrbench(data_root / "HR-Bench" / "hr_bench_4k.tsv", data_root, hr_image_dir, "4k")
        write_json(out_dir / "hrbench_4k.json", rows)
        summary["hrbench_4k"] = {"rows": len(rows), "path": str(out_dir / "hrbench_4k.json")}
    if args.hr_split in ("8k", "both"):
        rows = normalize_hrbench(data_root / "HR-Bench" / "hr_bench_8k.tsv", data_root, hr_image_dir, "8k")
        write_json(out_dir / "hrbench_8k.json", rows)
        summary["hrbench_8k"] = {"rows": len(rows), "path": str(out_dir / "hrbench_8k.json")}

    mme_src = data_root / "MME-RealWorld" / "MME_RealWorld.json"
    mme_rows, missing_images = normalize_mme(mme_src, data_root)
    if args.mme_max_samples and args.mme_max_samples > 0:
        mme_rows = mme_rows[: args.mme_max_samples]
    write_json(out_dir / "mme_realworld.json", mme_rows)
    summary["mme_realworld"] = {
        "rows": len(mme_rows),
        "path": str(out_dir / "mme_realworld.json"),
        "missing_images_skipped": missing_images,
    }

    write_json(out_dir / "summary.json", [{"dataset": k, **v} for k, v in summary.items()])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
