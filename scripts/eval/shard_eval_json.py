#!/usr/bin/env python3
"""Split an evaluation JSON list into balanced shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON list file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", required=True, help="Output filename prefix")
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected JSON list in {src}")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    summary = []
    total = len(rows)
    for shard_idx in range(args.num_shards):
        start = total * shard_idx // args.num_shards
        end = total * (shard_idx + 1) // args.num_shards
        shard_rows = rows[start:end]
        out_path = out_dir / f"{args.prefix}_shard{shard_idx:02d}_of{args.num_shards:02d}.json"
        out_path.write_text(json.dumps(shard_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(
            {
                "shard": shard_idx,
                "num_shards": args.num_shards,
                "start": start,
                "end": end,
                "rows": len(shard_rows),
                "path": str(out_path),
            }
        )

    summary_path = out_dir / f"{args.prefix}_shards_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(src), "rows": total, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

