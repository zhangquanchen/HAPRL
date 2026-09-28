#!/usr/bin/env python3
"""Merge shard-level eval metrics using shard row counts as weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict) -> float:
    for key in ("val/raw/qa_data/accuracy", "val/raw/iou_data/accuracy", "val/raw/accuracy"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"No accuracy key found in metrics: {sorted(metrics)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Shard summary JSON from shard_eval_json.py")
    parser.add_argument("--run-glob", required=True, help="Glob for shard run metrics, e.g. logs/runs/eval_x_shard*/val_trajs/metrics.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = read_json(Path(args.summary))
    rows_by_shard = {int(item["shard"]): int(item["rows"]) for item in summary}

    metrics_files = sorted(Path().glob(args.run_glob))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics matched {args.run_glob}")

    parts = []
    for path in metrics_files:
        name = str(path)
        shard = None
        for candidate in rows_by_shard:
            token = f"shard{candidate:02d}"
            if token in name:
                shard = candidate
                break
        if shard is None:
            raise ValueError(f"Cannot infer shard id from {path}")
        acc = metric_value(read_json(path))
        rows = rows_by_shard[shard]
        parts.append({"shard": shard, "rows": rows, "accuracy": acc, "metrics": str(path)})

    parts.sort(key=lambda item: item["shard"])
    total_rows = sum(item["rows"] for item in parts)
    weighted_acc = sum(item["accuracy"] * item["rows"] for item in parts) / total_rows
    result = {
        "rows": total_rows,
        "accuracy": weighted_acc,
        "parts": parts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

