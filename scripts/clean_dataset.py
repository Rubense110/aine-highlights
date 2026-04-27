#!/usr/bin/env python3
"""Clean Whisper auto-labels with text-based temporal non-maximum suppression."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


DEFAULT_INPUT_PATH = Path("data/whisper/dataset_whisper.json")
DEFAULT_OUTPUT_PATH = Path("data/whisper/dataset_cleaned.json")
DEFAULT_MAX_START_GAP_SECONDS = 10.0
DEFAULT_MIN_JACCARD = 0.20
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate overlapping transcript windows with text-based NMS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input dataset JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output cleaned JSON path.",
    )
    parser.add_argument(
        "--max-start-gap-seconds",
        type=float,
        default=DEFAULT_MAX_START_GAP_SECONDS,
        help="Consecutive windows within this start-time gap may belong to one event.",
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=DEFAULT_MIN_JACCARD,
        help="Minimum word Jaccard similarity to merge consecutive windows.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of dataset items in {path}")
    return payload


def save_dataset(path: Path, items: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(items), handle, indent=2, ensure_ascii=False)


def content_words(text: str) -> set[str]:
    """Lowercase words longer than 3 chars; enough for lightweight overlap."""
    return {
        match.group(0).lower()
        for match in WORD_RE.finditer(text)
        if len(match.group(0)) > 3
    }


def jaccard_similarity(left: str, right: str) -> float:
    left_words = content_words(left)
    right_words = content_words(right)
    if not left_words or not right_words:
        return 0.0
    intersection = len(left_words & right_words)
    union = len(left_words | right_words)
    return intersection / union if union else 0.0


def same_event(
    previous_item: dict[str, Any],
    current_item: dict[str, Any],
    max_start_gap_seconds: float,
    min_jaccard: float,
) -> bool:
    start_gap = float(current_item["start_s"]) - float(previous_item["start_s"])
    if start_gap > max_start_gap_seconds:
        return False
    similarity = jaccard_similarity(
        str(previous_item.get("text_query", "")),
        str(current_item.get("text_query", "")),
    )
    return similarity > min_jaccard


def choose_champion(cluster: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Keep the item with the richest transcript."""
    return max(
        cluster,
        key=lambda item: (
            len(str(item.get("text_query", "")).split()),
            len(str(item.get("text_query", ""))),
        ),
    )


def clean_dataset(
    items: Sequence[dict[str, Any]],
    max_start_gap_seconds: float,
    min_jaccard: float,
) -> list[dict[str, Any]]:
    if not items:
        return []

    sorted_items = sorted(items, key=lambda item: float(item["start_s"]))
    cleaned: list[dict[str, Any]] = []
    cluster: list[dict[str, Any]] = [sorted_items[0]]

    for current_item in sorted_items[1:]:
        previous_item = cluster[-1]
        if same_event(
            previous_item=previous_item,
            current_item=current_item,
            max_start_gap_seconds=max_start_gap_seconds,
            min_jaccard=min_jaccard,
        ):
            cluster.append(current_item)
        else:
            cleaned.append(choose_champion(cluster))
            cluster = [current_item]

    cleaned.append(choose_champion(cluster))
    return sorted(cleaned, key=lambda item: float(item["start_s"]))


def main() -> None:
    args = parse_args()
    items = load_dataset(args.input)
    cleaned = clean_dataset(
        items=items,
        max_start_gap_seconds=args.max_start_gap_seconds,
        min_jaccard=args.min_jaccard,
    )
    save_dataset(args.output, cleaned)
    print(f"Original windows: {len(items)} -> Cleaned windows: {len(cleaned)}")
    print(f"Saved cleaned dataset to {args.output}")


if __name__ == "__main__":
    main()
