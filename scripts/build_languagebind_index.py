#!/usr/bin/env python3
"""Build a persistent LanguageBind retrieval index for one gameplay video.

Console example:
    venv/bin/python scripts/build_languagebind_index.py \
        --video data/proxy_720p30.mp4 \
        --window-seconds 8 \
        --stride-seconds 2 \
        --batch-size 32 \
        --output-dir indexes/proxy_720p30_w8_s2

Notebook example:
    from scripts.build_languagebind_index import load_languagebind_index
    index = load_languagebind_index("indexes/proxy_720p30_w8_s2")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tfvtg_poc import (
    AUDIO_MODEL_ID,
    DEFAULT_BATCH_SIZE,
    DEFAULT_AUDIO_WEIGHT,
    DEFAULT_NUM_WORKERS,
    DEFAULT_STRIDE_SECONDS,
    DEFAULT_VIDEO_WEIGHT,
    DEFAULT_WINDOW_SECONDS,
    VIDEO_MODEL_ID,
    RetrievalResult,
    WindowSpec,
    extract_modal_embeddings_batched,
    extract_text_embedding,
    weighted_fuse_embeddings,
    load_audio_track,
    load_languagebind_models,
    make_windows,
    open_video_reader,
    print_results,
    rank_windows,
    require_cuda,
)


EMBEDDING_FILENAMES = {
    "video": "video_embeddings.pt",
    "audio": "audio_embeddings.pt",
    "scene": "scene_embeddings.pt",
}


@dataclass(frozen=True)
class LanguageBindIndex:
    index_dir: Path
    windows: list[WindowSpec]
    metadata: dict[str, Any]
    video_embeddings: torch.Tensor
    audio_embeddings: torch.Tensor
    scene_embeddings: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a persistent LanguageBind video moment retrieval index."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/proxy_720p30.mp4"),
        help="Input gameplay MP4.",
    )
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes for async video/audio preprocessing.",
    )
    parser.add_argument("--video-weight", type=float, default=DEFAULT_VIDEO_WEIGHT)
    parser.add_argument("--audio-weight", type=float, default=DEFAULT_AUDIO_WEIGHT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Index output directory. Defaults to indexes/<video_stem>_w<window>_s<stride>.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache_dir"),
        help="Hugging Face/model cache directory.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="On-disk embedding dtype. float16 is smaller and usually enough for cosine search.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty index directory.",
    )
    return parser.parse_args()


def default_index_dir(video_path: Path, window_seconds: float, stride_seconds: float) -> Path:
    window_tag = f"{window_seconds:g}".replace(".", "p")
    stride_tag = f"{stride_seconds:g}".replace(".", "p")
    return Path("indexes") / f"{video_path.stem}_w{window_tag}_s{stride_tag}"


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [path for path in output_dir.iterdir() if path.is_file()]
    if existing_files and not overwrite:
        raise FileExistsError(
            f"Index directory is not empty: {output_dir}. "
            "Pass --overwrite or choose another --output-dir."
        )
    return output_dir


def tensor_for_storage(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    tensor = tensor.detach().cpu().contiguous()
    if dtype == "float16":
        return tensor.to(torch.float16)
    if dtype == "float32":
        return tensor.to(torch.float32)
    raise ValueError(f"Unsupported dtype: {dtype}")


def save_windows(path: Path, windows: Sequence[WindowSpec]) -> None:
    payload = [asdict(window) for window in windows]
    path.write_text(json.dumps(payload, indent=2))


def load_windows(path: Path) -> list[WindowSpec]:
    payload = json.loads(path.read_text())
    return [WindowSpec(start_s=float(item["start_s"]), end_s=float(item["end_s"])) for item in payload]


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def build_languagebind_index(
    video_path: Path,
    output_dir: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    video_weight: float = DEFAULT_VIDEO_WEIGHT,
    audio_weight: float = DEFAULT_AUDIO_WEIGHT,
    cache_dir: Path = Path("cache_dir"),
    dtype: str = "float16",
    overwrite: bool = False,
) -> LanguageBindIndex:
    """Encode a video once and persist its LanguageBind retrieval index."""
    device = require_cuda()
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    cache_dir.mkdir(parents=True, exist_ok=True)

    (
        video_model,
        audio_model,
        video_processor,
        audio_processor,
        _text_tokenizer,
    ) = load_languagebind_models(cache_dir=cache_dir, device=device)

    video_reader, fps, duration_s = open_video_reader(video_path)
    audio_track = load_audio_track(video_path)
    windows = make_windows(
        duration_s=duration_s,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )
    if not windows:
        raise RuntimeError("No windows were produced from the input video.")

    print(
        f"Building index: video={video_path}, duration={duration_s:.2f}s, "
        f"fps={fps:.2f}, windows={len(windows)}, batch_size={batch_size}, "
        f"num_workers={num_workers}, "
        f"audio={'yes' if audio_track.has_audio else 'no/silence'}"
    )

    with torch.no_grad():
        video_embeddings, audio_embeddings, scene_embeddings = extract_modal_embeddings_batched(
            windows=windows,
            video_path=video_path,
            video_reader=video_reader,
            fps=fps,
            audio_track=audio_track,
            video_model=video_model,
            audio_model=audio_model,
            video_processor=video_processor,
            audio_processor=audio_processor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            video_weight=video_weight,
            audio_weight=audio_weight,
        )

    video_to_save = tensor_for_storage(video_embeddings, dtype)
    audio_to_save = tensor_for_storage(audio_embeddings, dtype)
    scene_to_save = tensor_for_storage(scene_embeddings, dtype)

    torch.save(video_to_save, output_dir / EMBEDDING_FILENAMES["video"])
    torch.save(audio_to_save, output_dir / EMBEDDING_FILENAMES["audio"])
    torch.save(scene_to_save, output_dir / EMBEDDING_FILENAMES["scene"])
    save_windows(output_dir / "windows.json", windows)

    video_stat = video_path.stat()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_path": str(video_path),
        "video_name": video_path.name,
        "video_size_bytes": video_stat.st_size,
        "video_mtime_ns": video_stat.st_mtime_ns,
        "duration_s": duration_s,
        "fps": fps,
        "num_video_frames": len(video_reader),
        "has_audio": audio_track.has_audio,
        "audio_sample_rate": audio_track.sample_rate,
        "num_windows": len(windows),
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "video_weight": video_weight,
        "audio_weight": audio_weight,
        "embedding_dtype": dtype,
        "video_model_id": VIDEO_MODEL_ID,
        "audio_model_id": AUDIO_MODEL_ID,
        "embedding_files": EMBEDDING_FILENAMES,
    }
    save_metadata(output_dir / "metadata.json", metadata)

    print(f"Saved index to {output_dir.resolve()}")
    print(f"scene_embeddings: {tuple(scene_to_save.shape)} {scene_to_save.dtype}")

    return LanguageBindIndex(
        index_dir=output_dir,
        windows=list(windows),
        metadata=metadata,
        video_embeddings=video_to_save,
        audio_embeddings=audio_to_save,
        scene_embeddings=scene_to_save,
    )


def load_languagebind_index(index_dir: str | Path, map_location: str | torch.device = "cpu") -> LanguageBindIndex:
    """Load a persisted index for notebook or script querying."""
    index_dir = Path(index_dir)
    metadata = json.loads((index_dir / "metadata.json").read_text())
    windows = load_windows(index_dir / "windows.json")
    video_embeddings = torch.load(index_dir / EMBEDDING_FILENAMES["video"], map_location=map_location)
    audio_embeddings = torch.load(index_dir / EMBEDDING_FILENAMES["audio"], map_location=map_location)
    scene_embeddings = torch.load(index_dir / EMBEDDING_FILENAMES["scene"], map_location=map_location)
    return LanguageBindIndex(
        index_dir=index_dir,
        windows=windows,
        metadata=metadata,
        video_embeddings=video_embeddings,
        audio_embeddings=audio_embeddings,
        scene_embeddings=scene_embeddings,
    )


def query_languagebind_index(
    index: LanguageBindIndex,
    text_model: torch.nn.Module,
    text_tokenizer: object,
    query: str,
    device: torch.device,
    top_k: int = 3,
    modality: str = "scene",
    video_weight: float = DEFAULT_VIDEO_WEIGHT,
    audio_weight: float = DEFAULT_AUDIO_WEIGHT,
) -> list[RetrievalResult]:
    """Rank a loaded index for one text query.

    modality can be "scene", "video", or "audio" for diagnostics.
    """
    if modality == "scene":
        video_embeddings = F.normalize(index.video_embeddings.to(device=device, dtype=torch.float32), dim=-1)
        audio_embeddings = F.normalize(index.audio_embeddings.to(device=device, dtype=torch.float32), dim=-1)
        embeddings = weighted_fuse_embeddings(
            video_embeddings,
            audio_embeddings,
            video_weight=video_weight,
            audio_weight=audio_weight,
        )
    elif modality == "stored_scene":
        embeddings = index.scene_embeddings
    elif modality == "video":
        embeddings = index.video_embeddings
    elif modality == "audio":
        embeddings = index.audio_embeddings
    else:
        raise ValueError("modality must be one of: scene, stored_scene, video, audio")

    embeddings = F.normalize(embeddings.to(device=device, dtype=torch.float32), dim=-1)
    with torch.no_grad():
        text_embedding = extract_text_embedding(text_model, text_tokenizer, query, device)
        return rank_windows(index.windows, embeddings, text_embedding, top_k)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_index_dir(args.video, args.window_seconds, args.stride_seconds)

    build_languagebind_index(
        video_path=args.video,
        output_dir=output_dir,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        video_weight=args.video_weight,
        audio_weight=args.audio_weight,
        cache_dir=args.cache_dir,
        dtype=args.dtype,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
