#!/usr/bin/env python3
"""Filter Mini-o3 cold-start data by assistant generation turns."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-assistant-turns", default=6, type=int)
    parser.add_argument("--input-file", default="Mini-o3-Coldstart.json")
    parser.add_argument("--dataset-info", default="dataset_info.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def assistant_turn_count(example: dict) -> int:
    conversations = example.get("conversations", [])
    return sum(1 for item in conversations if item.get("from") in {"gpt", "assistant"})


def main() -> None:
    args = parse_args()
    input_json = args.input_dir / args.input_file
    input_dataset_info = args.input_dir / args.dataset_info
    output_json = args.output_dir / args.input_file
    output_dataset_info = args.output_dir / args.dataset_info
    metadata_json = args.output_dir / "filter_metadata.json"

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"[fatal] output dir is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_json.read_text())
    turn_hist = Counter(assistant_turn_count(example) for example in data)
    kept = [example for example in data if assistant_turn_count(example) <= args.max_assistant_turns]
    kept_hist = Counter(assistant_turn_count(example) for example in kept)

    output_json.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    if input_dataset_info.exists():
        shutil.copy2(input_dataset_info, output_dataset_info)

    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "max_assistant_turns": args.max_assistant_turns,
        "total_examples": len(data),
        "kept_examples": len(kept),
        "dropped_examples": len(data) - len(kept),
        "keep_ratio": len(kept) / len(data) if data else 0.0,
        "turn_hist": dict(sorted(turn_hist.items())),
        "kept_turn_hist": dict(sorted(kept_hist.items())),
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {output_json}")
    if input_dataset_info.exists():
        print(f"[ok] copied {output_dataset_info}")


if __name__ == "__main__":
    main()
