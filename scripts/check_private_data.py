#!/usr/bin/env python3
"""Scan this package for machine paths, credentials, or leftover result files.

Run it before handing the package to anyone. A clean run only means no known
pattern matched; it does not replace a manual privacy and licence review.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".env",
    ".ini",
    ".ipynb",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".out",
    ".py",
    ".rst",
    ".sh",
    ".slurm",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
}

# These files define the detection patterns themselves, so they would always
# match. They are reviewed by reading them, not by scanning them.
DETECTOR_FILES = {
    "scripts/check_private_data.py",
    "data/verify_data.py",
}

FORBIDDEN_FILE_PATTERNS = (
    re.compile(r"(^|[_-])metrics?(\.|[_-])", re.IGNORECASE),
    re.compile(r"(^|[_-])results?(\.|[_-])", re.IGNORECASE),
    re.compile(r"tensorboard|events\.out\.tfevents", re.IGNORECASE),
    re.compile(r"slurm-\S+\.(out|err)$", re.IGNORECASE),
)

CONTENT_RULES = {
    "macOS absolute user path": re.compile(r"/Users/[^/\s]+/"),
    "Linux absolute home path": re.compile(r"/home/[^/\s]+/"),
    "cluster user-data path": re.compile(r"/data/(?:user|users|home)/[^/\s]+/", re.IGNORECASE),
    "mounted absolute path": re.compile(r"/mnt/[^\s'\"`]+", re.IGNORECASE),
    "Windows user path": re.compile(
        r"\b[A-Z]:(?:\\|/)Users(?:\\|/)[^\\/\s]+(?:\\|/)",
        re.IGNORECASE,
    ),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "private or internal IPv4 address": re.compile(
        r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|"
        r"192\.168\.(?:\d{1,3}\.)\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3}|"
        r"127\.(?:\d{1,3}\.){2}\d{1,3}|"
        r"169\.254\.(?:\d{1,3}\.)\d{1,3}|"
        r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.(?:\d{1,3}\.)\d{1,3})\b"
    ),
    "SSH/SCP host path": re.compile(r"\b(?:ssh|scp|rsync)\s+[^#\n]*\b[A-Za-z0-9._-]+@", re.IGNORECASE),
    "scheduler account": re.compile(r"(?:#SBATCH\s+(?:--account|-A)|SBATCH_ACCOUNT\s*=)\s*\S+", re.IGNORECASE),
    "populated bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    "populated secret assignment": re.compile(
        r"(?:API[_-]?KEY|ACCESS[_-]?TOKEN|SECRET|PASSWORD)\s*[:=]\s*"
        r"(?!(?:REPLACE|YOUR_|<|\$\{|['\"]?\s*$))[A-Za-z0-9._~+/=-]{12,}",
        re.IGNORECASE,
    ),
    "evaluation percentage": re.compile(
        r"(?:accuracy|precision|recall|f1|auc|pass@|mean[_ -]?iou|macro[_ -]?average)"
        r"[^%\n]{0,48}\b\d+(?:\.\d+)?\s*%",
        re.IGNORECASE,
    ),
    "result-table row": re.compile(
        r"^\s*\|[^\n|]*(?:accuracy|precision|recall|f1|auc|mean[_ -]?iou)[^\n|]*\|"
        r"[^\n]*\|\s*\d+(?:\.\d+)?\s*\|",
        re.IGNORECASE | re.MULTILINE,
    ),
    "historical training metric log": re.compile(
        r"^[^\n]{0,100}\b(?:step|global_step)\s*[:=]?\s*\d+"
        r"[^\n]{0,300}\b(?:critic|actor)(?:[/_.-](?:loss|reward|score|entropy|grad_norm|kl))\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    "executed notebook output": re.compile(r'"outputs"\s*:\s*\[\s*\{', re.IGNORECASE),
}


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name.endswith(".env.example")


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue

        relative = path.relative_to(root)
        if relative.as_posix() in DETECTOR_FILES:
            continue

        for pattern in FORBIDDEN_FILE_PATTERNS:
            if pattern.search(path.name):
                findings.append(f"{relative}: forbidden result/artifact filename")
                break

        if not is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for label, pattern in CONTENT_RULES.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {label}")

    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[fatal] not a directory: {root}", file=sys.stderr)
        return 2

    findings = scan(root)
    if findings:
        print("[failed] possible private data or leftover results:", file=sys.stderr)
        for item in findings:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("[ok] no known machine-path, credential, or result patterns found")
    print("[note] a manual privacy and licence review is still required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

