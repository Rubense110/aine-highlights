#!/usr/bin/env python3
"""Merge cleaned Whisper and OCR datasets into one master dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_WHISPER_PATH = Path("data/whisper/dataset_cleaned.json")
DEFAULT_OCR_PATH = Path("data/ocr/dataset_ocr_cleaned.json")
DEFAULT_OUTPUT_PATH = Path("data/master_dataset_raw.json")
DEFAULT_TIME_TOLERANCE_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge cleaned Whisper transcript labels and OCR HUD labels."
    )
    parser.add_argument(
        "--whisper",
        type=Path,
        default=DEFAULT_WHISPER_PATH,
        help="Cleaned Whisper dataset JSON path.",
    )
    parser.add_argument(
        "--ocr",
        type=Path,
        default=DEFAULT_OCR_PATH,
        help="Cleaned OCR dataset JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Merged master dataset JSON path.",
    )
    parser.add_argument(
        "--time-tolerance-seconds",
        type=float,
        default=DEFAULT_TIME_TOLERANCE_SECONDS,
        help="Merge items whose start times differ by no more than this many seconds.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def save_dataset(path: Path, items: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(items), handle, indent=2, ensure_ascii=False)


def format_audio_text(text: str) -> str:
    return f'[Audio: "{text}"]'


def format_hud_text(text: str) -> str:
    return f'[HUD: "{text}"]'


def merge_pair(whisper_item: dict[str, Any], ocr_item: dict[str, Any]) -> dict[str, Any]:
    start_s = min(float(whisper_item["start_s"]), float(ocr_item["start_s"]))
    end_s = max(float(whisper_item["end_s"]), float(ocr_item["end_s"]))
    return {
        "window_index": int(whisper_item["window_index"]),
        "start_s": start_s,
        "end_s": end_s,
        "text_query": (
            f'{format_audio_text(str(whisper_item.get("text_query", "")))} '
            f'{format_hud_text(str(ocr_item.get("text_query", "")))}'
        ),
    }


def whisper_only(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_index": int(item["window_index"]),
        "start_s": float(item["start_s"]),
        "end_s": float(item["end_s"]),
        "text_query": format_audio_text(str(item.get("text_query", ""))),
    }


def ocr_only(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_index": int(item["window_index"]),
        "start_s": float(item["start_s"]),
        "end_s": float(item["end_s"]),
        "text_query": format_hud_text(str(item.get("text_query", ""))),
    }


def find_matching_ocr_index(
    whisper_item: dict[str, Any],
    ocr_items: Sequence[dict[str, Any]],
    used_ocr_indices: set[int],
    time_tolerance_seconds: float,
) -> int | None:
    whisper_window_index = int(whisper_item["window_index"])
    whisper_start_s = float(whisper_item["start_s"])

    best_index: int | None = None
    best_gap: float | None = None
    for ocr_index, ocr_item in enumerate(ocr_items):
        if ocr_index in used_ocr_indices:
            continue
        same_window = int(ocr_item["window_index"]) == whisper_window_index
        ocr_start_s = float(ocr_item["start_s"])
        reaction_delay_s = whisper_start_s - ocr_start_s
        same_event = same_window or (-2.0 <= reaction_delay_s <= 8.0)
        if not same_event:
            continue
        start_gap = abs(reaction_delay_s)
        if best_gap is None or start_gap < best_gap:
            best_index = ocr_index
            best_gap = start_gap
    return best_index


def merge_datasets(
    whisper_items: Sequence[dict[str, Any]],
    ocr_items: Sequence[dict[str, Any]],
    time_tolerance_seconds: float,
) -> list[dict[str, Any]]:
    whisper_sorted = sorted(whisper_items, key=lambda item: float(item["start_s"]))
    ocr_sorted = sorted(ocr_items, key=lambda item: float(item["start_s"]))
    used_ocr_indices: set[int] = set()
    merged: list[dict[str, Any]] = []

    for whisper_item in whisper_sorted:
        ocr_index = find_matching_ocr_index(
            whisper_item=whisper_item,
            ocr_items=ocr_sorted,
            used_ocr_indices=used_ocr_indices,
            time_tolerance_seconds=time_tolerance_seconds,
        )
        if ocr_index is None:
            merged.append(whisper_only(whisper_item))
            continue

        used_ocr_indices.add(ocr_index)
        merged.append(merge_pair(whisper_item, ocr_sorted[ocr_index]))

    for ocr_index, ocr_item in enumerate(ocr_sorted):
        if ocr_index not in used_ocr_indices:
            merged.append(ocr_only(ocr_item))

    return sorted(merged, key=lambda item: float(item["start_s"]))


def main() -> None:
    args = parse_args()
    whisper_items = load_dataset(args.whisper)
    ocr_items = load_dataset(args.ocr)
    merged = merge_datasets(
        whisper_items=whisper_items,
        ocr_items=ocr_items,
        time_tolerance_seconds=args.time_tolerance_seconds,
    )
    save_dataset(args.output, merged)
    print(
        f"Whisper: {len(whisper_items)} + OCR: {len(ocr_items)} -> "
        f"Master: {len(merged)}"
    )
    print(f"Saved master dataset to {args.output}")


if __name__ == "__main__":
    main()
