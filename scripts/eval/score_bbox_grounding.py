#!/usr/bin/env python3
"""Score VStar grounding rollouts with independently computed bbox IoU."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)
BBOX_RE = re.compile(r"<bbox>\s*([0-9eE\.\,\-\s\[\]]+)\s*</bbox>", flags=re.DOTALL | re.IGNORECASE)
BRACKET_RE = re.compile(
    r"\[\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*\]"
)
GROUNDING_RE = re.compile(r"<grounding>(.*?)</grounding>", flags=re.DOTALL | re.IGNORECASE)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_nums(payload: str) -> list[float] | None:
    payload = payload.strip().strip("[]")
    parts = [p for p in re.split(r"[,\s]+", payload) if p]
    if len(parts) != 4:
        return None
    try:
        vals = [float(p) for p in parts]
    except Exception:
        return None
    return vals


def sanitize_bbox(vals: list[float] | None) -> list[float] | None:
    if vals is None or len(vals) != 4:
        return None
    x0, y0, x1, y1 = vals
    vals = [max(0.0, min(1.0, float(v))) for v in (x0, y0, x1, y1)]
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def parse_final_answer_bbox(text: str) -> tuple[list[float] | None, str]:
    """Parse bbox only from the final answer block for the strict metric."""
    if not isinstance(text, str):
        return None, "not_string"
    answer_blocks = ANSWER_RE.findall(text)
    if not answer_blocks:
        return None, "missing_answer"
    answer = answer_blocks[-1]
    bbox_matches = BBOX_RE.findall(answer)
    if bbox_matches:
        box = sanitize_bbox(parse_nums(bbox_matches[-1]))
        return (box, "answer_bbox") if box is not None else (None, "invalid_answer_bbox")
    bracket_matches = BRACKET_RE.findall(answer)
    if bracket_matches:
        box = sanitize_bbox([float(x) for x in bracket_matches[-1]])
        return (box, "answer_bracket") if box is not None else (None, "invalid_answer_bracket")
    return None, "missing_bbox_in_answer"


def parse_grounding_payload(payload: str) -> tuple[list[float], str] | None:
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    bbox = sanitize_bbox(obj.get("bbox_2d"))
    source = str(obj.get("source", "original_image"))
    if bbox is None:
        return None
    return bbox, source


def map_child_to_parent(child: list[float], parent: list[float]) -> list[float] | None:
    px0, py0, px1, py1 = parent
    pw = px1 - px0
    ph = py1 - py0
    if pw <= 0 or ph <= 0:
        return None
    mapped = [
        px0 + child[0] * pw,
        py0 + child[1] * ph,
        px0 + child[2] * pw,
        py0 + child[3] * ph,
    ]
    return sanitize_bbox(mapped)


def parse_last_grounding_as_original_bbox(text: str) -> list[float] | None:
    """Map the last valid grounding bbox back to original-image coordinates."""
    obs_boxes: list[list[float]] = []
    last: list[float] | None = None
    for payload in GROUNDING_RE.findall(text or ""):
        parsed = parse_grounding_payload(payload.strip())
        if parsed is None:
            continue
        bbox, source = parsed
        if source == "original_image":
            orig = bbox
        elif source.startswith("observation_"):
            try:
                idx = int(source.rsplit("_", 1)[-1]) - 1
            except Exception:
                continue
            if idx < 0 or idx >= len(obs_boxes):
                continue
            orig = map_child_to_parent(bbox, obs_boxes[idx])
            if orig is None:
                continue
        else:
            continue
        obs_boxes.append(orig)
        last = orig
    return last


def iou(a: list[float], b: list[float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def max_iou(pred: list[float] | None, gt_boxes: list[list[float]]) -> float:
    if pred is None or not gt_boxes:
        return 0.0
    return max(iou(pred, gt) for gt in gt_boxes)


def gt_boxes_from_row(row: dict[str, Any]) -> list[list[float]]:
    extra = row.get("extra_info") or {}
    boxes = extra.get("gt_bboxes_norm") or []
    out = []
    for box in boxes:
        clean = sanitize_bbox(box)
        if clean is not None:
            out.append(clean)
    if out:
        return out
    # Fallback for older prepared files with only solution=<bbox>...</bbox>.
    m = BBOX_RE.findall(str(row.get("solution", "")))
    for item in m:
        clean = sanitize_bbox(parse_nums(item))
        if clean is not None:
            out.append(clean)
    return out


def extract_output(rollout: Any) -> str:
    if isinstance(rollout, dict):
        return str(rollout.get("output", ""))
    return str(rollout)


def pct(x: float) -> float:
    return round(100.0 * x, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-json", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--rollout-index", type=int, default=0)
    args = parser.parse_args()

    prepared = read_json(Path(args.prepared_json))
    stats = read_json(Path(args.stats_json))
    prepared_by_doc = {row["doc_id"]: row for row in prepared}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in stats:
        doc_id = item.get("doc_id")
        prep = prepared_by_doc.get(doc_id)
        gt_boxes = gt_boxes_from_row(prep or {})
        rollouts = item.get("rollouts") or []
        if args.rollout_index < len(rollouts):
            output = extract_output(rollouts[args.rollout_index])
        elif rollouts:
            output = extract_output(rollouts[0])
        else:
            output = ""
        pred, parse_source = parse_final_answer_bbox(output)
        strict_iou = max_iou(pred, gt_boxes)
        fallback_pred = parse_last_grounding_as_original_bbox(output)
        fallback_iou = max_iou(fallback_pred, gt_boxes)
        extra = (prep or {}).get("extra_info") or {}
        rows.append(
            {
                "model_key": args.model_key,
                "checkpoint": args.checkpoint,
                "doc_id": doc_id,
                "category": extra.get("category", ""),
                "question_id": extra.get("question_id", ""),
                "parse_ok": pred is not None,
                "parse_source": parse_source,
                "iou": strict_iou,
                "iou50": strict_iou >= args.iou_threshold,
                "gt_bboxes": json.dumps(gt_boxes, ensure_ascii=False),
                "pred_bbox": json.dumps(pred, ensure_ascii=False),
                "fallback_grounding_bbox": json.dumps(fallback_pred, ensure_ascii=False),
                "fallback_grounding_iou": fallback_iou,
                "fallback_grounding_iou50": fallback_iou >= args.iou_threshold,
                "output": output,
            }
        )

    n = len(rows)
    parse_ok = sum(1 for r in rows if r["parse_ok"])
    iou50 = sum(1 for r in rows if r["iou50"])
    fallback_iou50 = sum(1 for r in rows if r["fallback_grounding_iou50"])
    ious = [float(r["iou"]) for r in rows]
    valid_ious = [float(r["iou"]) for r in rows if r["parse_ok"]]
    fallback_ious = [float(r["fallback_grounding_iou"]) for r in rows]
    summary = {
        "model_key": args.model_key,
        "checkpoint": args.checkpoint,
        "samples": n,
        "parse_ok": parse_ok,
        "parse_rate": parse_ok / n if n else 0.0,
        "parse_rate_%": pct(parse_ok / n) if n else 0.0,
        "mean_iou": sum(ious) / n if n else 0.0,
        "mean_iou_%": pct(sum(ious) / n) if n else 0.0,
        "mean_iou_valid": sum(valid_ious) / len(valid_ious) if valid_ious else 0.0,
        "mean_iou_valid_%": pct(sum(valid_ious) / len(valid_ious)) if valid_ious else 0.0,
        "median_iou": statistics.median(ious) if ious else 0.0,
        "iou50_correct": iou50,
        "iou50_acc": iou50 / n if n else 0.0,
        "iou50_acc_%": pct(iou50 / n) if n else 0.0,
        "fallback_grounding_mean_iou": sum(fallback_ious) / n if n else 0.0,
        "fallback_grounding_mean_iou_%": pct(sum(fallback_ious) / n) if n else 0.0,
        "fallback_grounding_iou50_acc": fallback_iou50 / n if n else 0.0,
        "fallback_grounding_iou50_acc_%": pct(fallback_iou50 / n) if n else 0.0,
        "iou_threshold": args.iou_threshold,
        "prepared_json": args.prepared_json,
        "stats_json": args.stats_json,
    }

    per_sample_jsonl = out_dir / "vstar_bbox_grounding_per_sample.jsonl"
    with per_sample_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_sample_csv = out_dir / "vstar_bbox_grounding_per_sample.csv"
    fieldnames = [k for k in rows[0].keys() if k != "output"] + ["output"] if rows else []
    if fieldnames:
        with per_sample_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary_json = out_dir / "vstar_bbox_grounding_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_csv = out_dir / "vstar_bbox_grounding_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
