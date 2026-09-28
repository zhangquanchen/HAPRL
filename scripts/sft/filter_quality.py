#!/usr/bin/env python3
"""Quality filter for Mini-o3 cold-start LLaMA-Factory data.

The filter is deliberately conservative: it removes examples with many
failure/recovery phrases or repeated full sentences, while keeping complex
multi-step examples that only look repetitive because of structured labels.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


RECOVERY_PHRASES = (
    "go back",
    "previous attempt",
    "previous attempts",
    "re-examine",
    "try a new",
    "better bounding box",
    "failed",
    "too low",
    "too high",
    "too far",
    "too blurry",
    "blurry",
    "unsuccessful",
    "not successful",
    "not enough",
    "incorrect",
    "misaligned",
)

SENTENCE_RE = re.compile(r"[^.!?\n。！？]+[.!?。！？]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-file", default="Mini-o3-Coldstart.json")
    parser.add_argument("--dataset-info", default="dataset_info.json")
    parser.add_argument("--max-recovery-phrase-hits", default=4, type=int)
    parser.add_argument("--max-sentence-repeat", default=1, type=int)
    parser.add_argument("--min-keep", default=5000, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def assistant_texts(example: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for item in example.get("conversations", []):
        if item.get("from") in {"gpt", "assistant"}:
            value = item.get("value", "")
            if isinstance(value, str):
                texts.append(value)
    return texts


def assistant_turn_count(example: dict[str, Any]) -> int:
    return len(assistant_texts(example))


def phrase_hits(text: str) -> Counter[str]:
    lowered = text.lower()
    return Counter({phrase: lowered.count(phrase) for phrase in RECOVERY_PHRASES if phrase in lowered})


def repeated_sentence_count(texts: list[str]) -> int:
    counts: Counter[str] = Counter()
    for text in texts:
        for match in SENTENCE_RE.findall(text):
            normalized = " ".join(match.lower().split())
            if len(normalized) >= 80:
                counts[normalized] += 1
    return sum(count - 1 for count in counts.values() if count > 1)


def final_has_answer(example: dict[str, Any]) -> bool:
    texts = assistant_texts(example)
    return bool(texts and "<answer>" in texts[-1] and "</answer>" in texts[-1])


def example_quality(example: dict[str, Any]) -> dict[str, Any]:
    texts = assistant_texts(example)
    joined = "\n".join(texts)
    hits = phrase_hits(joined)
    return {
        "assistant_turns": len(texts),
        "recovery_phrase_hits": sum(hits.values()),
        "recovery_phrase_detail": dict(hits),
        "sentence_repeat": repeated_sentence_count(texts),
        "final_has_answer": final_has_answer(example),
        "chars": len(joined),
    }


def should_keep(quality: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not quality["final_has_answer"]:
        reasons.append("missing_final_answer")
    if quality["recovery_phrase_hits"] > args.max_recovery_phrase_hits:
        reasons.append("too_many_recovery_phrases")
    if quality["sentence_repeat"] > args.max_sentence_repeat:
        reasons.append("repeated_sentences")
    return not reasons, reasons


def main() -> None:
    args = parse_args()
    input_json = args.input_dir / args.input_file
    input_dataset_info = args.input_dir / args.dataset_info
    output_json = args.output_dir / args.input_file
    output_dataset_info = args.output_dir / args.dataset_info
    metadata_json = args.output_dir / "filter_quality_metadata.json"

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"[fatal] output dir is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_json.read_text())
    kept: list[dict[str, Any]] = []
    dropped_examples: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    phrase_counter: Counter[str] = Counter()
    turn_hist: Counter[int] = Counter()
    kept_turn_hist: Counter[int] = Counter()

    for idx, example in enumerate(data):
        quality = example_quality(example)
        keep, reasons = should_keep(quality, args)
        turn_hist[quality["assistant_turns"]] += 1
        phrase_counter.update(quality["recovery_phrase_detail"])
        if keep:
            kept.append(example)
            kept_turn_hist[quality["assistant_turns"]] += 1
        else:
            reason_counter.update(reasons)
            dropped_examples.append(
                {
                    "idx": idx,
                    "id": example.get("id") or example.get("uid") or example.get("doc_id"),
                    "reasons": reasons,
                    "assistant_turns": quality["assistant_turns"],
                    "recovery_phrase_hits": quality["recovery_phrase_hits"],
                    "sentence_repeat": quality["sentence_repeat"],
                    "chars": quality["chars"],
                }
            )

    if len(kept) < args.min_keep:
        raise SystemExit(
            f"[fatal] only kept {len(kept)} examples, below --min-keep={args.min_keep}; "
            "relax thresholds before training"
        )

    output_json.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    if input_dataset_info.exists():
        shutil.copy2(input_dataset_info, output_dataset_info)

    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_file": args.input_file,
        "max_recovery_phrase_hits": args.max_recovery_phrase_hits,
        "max_sentence_repeat": args.max_sentence_repeat,
        "min_keep": args.min_keep,
        "total_examples": len(data),
        "kept_examples": len(kept),
        "dropped_examples": len(data) - len(kept),
        "keep_ratio": len(kept) / len(data) if data else 0.0,
        "drop_reasons": dict(reason_counter),
        "turn_hist": dict(sorted(turn_hist.items())),
        "kept_turn_hist": dict(sorted(kept_turn_hist.items())),
        "top_recovery_phrases": phrase_counter.most_common(20),
        "dropped_preview": dropped_examples[:50],
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {output_json}")
    print(f"[ok] wrote {metadata_json}")
    if input_dataset_info.exists():
        print(f"[ok] copied {output_dataset_info}")


if __name__ == "__main__":
    main()
