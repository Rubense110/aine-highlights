#!/usr/bin/env python3
"""Query a persisted LanguageBind index from the console."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_languagebind_index import load_languagebind_index, query_languagebind_index
from scripts.tfvtg_poc import (
    DEFAULT_AUDIO_WEIGHT,
    DEFAULT_CLIP_CONTEXT_SECONDS,
    DEFAULT_VIDEO_WEIGHT,
    export_result_clips,
    load_languagebind_models,
    print_results,
    require_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a persisted LanguageBind retrieval index.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--modality",
        choices=("scene", "stored_scene", "video", "audio"),
        default="scene",
        help="Rank weighted scene, stored_scene, video-only, or audio-only embeddings.",
    )
    parser.add_argument("--video-weight", type=float, default=DEFAULT_VIDEO_WEIGHT)
    parser.add_argument("--audio-weight", type=float, default=DEFAULT_AUDIO_WEIGHT)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache_dir"))
    parser.add_argument("--export-clips", action="store_true")
    parser.add_argument("--clip-output-dir", type=Path, default=Path("outputs/validation_clips"))
    parser.add_argument("--clip-context-seconds", type=float, default=DEFAULT_CLIP_CONTEXT_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_cuda()
    index = load_languagebind_index(args.index_dir)

    video_model, _audio_model, _video_processor, _audio_processor, text_tokenizer = load_languagebind_models(
        cache_dir=args.cache_dir,
        device=device,
    )

    results = query_languagebind_index(
        index=index,
        text_model=video_model,
        text_tokenizer=text_tokenizer,
        query=args.query,
        device=device,
        top_k=args.top_k,
        modality=args.modality,
        video_weight=args.video_weight,
        audio_weight=args.audio_weight,
    )

    if args.export_clips:
        video_path = Path(index.metadata["video_path"])
        results = export_result_clips(
            video_path=video_path,
            results=results,
            output_dir=args.clip_output_dir,
            query=args.query,
            context_seconds=args.clip_context_seconds,
            max_duration_s=float(index.metadata["duration_s"]),
        )
    print_results(results)


if __name__ == "__main__":
    main()
