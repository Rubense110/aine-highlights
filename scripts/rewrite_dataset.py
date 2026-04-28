#!/usr/bin/env python3
"""Rewrite raw master dataset labels with a local Ollama model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm


DEFAULT_INPUT_PATH = Path("data/master_dataset_raw.json")
DEFAULT_OUTPUT_PATH = Path("data/master_dataset_final.json")
DEFAULT_MODEL = "llama3.1"
SYSTEM_PROMPT = (
    "You are an expert video gameplay annotator for Helldivers 2. Your job is to translate "
    "raw player transcripts and HUD tags into a rich, descriptive English sentence. "
    "CRITICAL RULES: "
    "1. Do not just translate literally. Deduce the context. "
    "2. Actively detect and describe player emotions and situations (e.g., panic, frustration, "
    "hysterical laughter, chaos, friendly fire, accidental deaths). "
    "3. Combine the HUD tags and the audio vibe into one cohesive scene description. "
    "4. Use the official Helldivers 2 glossary and avoid generic game-mechanic wording: "
    "write 'Hellpod' instead of 'capsule', 'pod', or 'drop'; write 'Stratagem' instead "
    "of 'tactical strategy' or 'plan'; if 'ÁGUILA' or 'EAGLE' appears, write "
    "'Eagle Airstrike'; if 'ORBITAL' appears, write 'Orbital Strike'; write 'Pelican-1' "
    "instead of 'rescue ship' or 'plane'. "
    "EXAMPLE 1: '[Audio: ¿Cómo se tiraba la granada o algo?!] [HUD: BRECHA DE BICHOS]' -> "
    "'A player panics and frantically asks for controls while being overwhelmed by a massive bug breach.' "
    "EXAMPLE 2: '[Audio: Me ha quemado con el motor, tío, vienen a rescatarte y te matan] [HUD: ELIMINADO]' -> "
    "'A player expresses heavy frustration after being accidentally incinerated by Pelican-1 during extraction.' "
    "EXAMPLE 3: '[Audio: Ya vuelvo, caigo encima de ellos] [HUD: REFUERZOS LISTOS]' -> "
    "'A player returns to the battlefield in a Hellpod descent as reinforcements become available.' "
    "EXAMPLE 4: '[Audio: Tira águila ahí, limpia la brecha] [HUD: ÁGUILA, BRECHA DE BICHOS]' -> "
    "'A player calls in an Eagle Airstrike Stratagem to clear a massive bug breach.' "
    "Output ONLY the English description."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite raw master labels into English descriptions with Ollama."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input master raw dataset JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output final dataset JSON path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to rewrite for testing.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def save_dataset(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2, ensure_ascii=False)


def clean_llm_output(text: str) -> str:
    """Normalize model output to one compact sentence."""
    cleaned = " ".join(text.strip().strip('"').strip("'").split())
    return cleaned


def rewrite_with_ollama(model: str, raw_text: str) -> str:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "The Ollama Python package is not installed. Run: venv/bin/pip install ollama"
        ) from exc

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Rewrite this raw annotation as one objective English video "
                    f"description sentence:\n{raw_text}"
                ),
            },
        ],
        options={
            "temperature": 0.2,
        },
    )
    return clean_llm_output(response["message"]["content"])


def rewrite_dataset(
    rows: Sequence[dict[str, Any]],
    model: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    rewritten_rows: list[dict[str, Any]] = []
    total = len(rows) if limit is None else min(len(rows), limit)

    for row_index, row in enumerate(tqdm(rows, total=len(rows), desc="Rewriting labels")):
        if limit is not None and row_index >= limit:
            rewritten_rows.append(dict(row))
            continue

        updated_row = dict(row)
        raw_text = str(row.get("text_query", ""))
        try:
            rewritten_text = rewrite_with_ollama(model=model, raw_text=raw_text)
        except Exception as exc:
            print(
                f"Warning: failed to rewrite row {row_index + 1}/{total} "
                f"(window_index={row.get('window_index')}): {exc}"
            )
            rewritten_rows.append(updated_row)
            continue

        if rewritten_text:
            updated_row["text_query"] = rewritten_text
        rewritten_rows.append(updated_row)

    return rewritten_rows


def main() -> None:
    args = parse_args()
    rows = load_dataset(args.input)
    rewritten_rows = rewrite_dataset(
        rows=rows,
        model=args.model,
        limit=args.limit,
    )
    save_dataset(args.output, rewritten_rows)
    print(f"Saved final dataset to {args.output}")


if __name__ == "__main__":
    main()
