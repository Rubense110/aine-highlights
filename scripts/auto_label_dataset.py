#!/usr/bin/env python3
"""Auto-label gameplay video windows with a volume -> Whisper cascade.

This script builds a small text-video fine-tuning dataset from one gameplay
video. It first keeps only loud/active windows using a cheap PyTorch RMS
heuristic, then sends the survivors through Whisper for transcription.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tfvtg_poc import (  # noqa: E402
    DEFAULT_STRIDE_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    AudioTrack,
    WindowSpec,
    load_audio_track,
    make_windows,
    open_video_reader,
    require_cuda,
)


DEFAULT_KEEP_TOP_PCT = 0.15
DEFAULT_LANGUAGE = "spanish"
DEFAULT_WHISPER_BATCH_SIZE = 8
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_INITIAL_PROMPT = (
    "Audio informal de tres amigos andaluces jugando al videojuego Helldivers 2. "
    "Se escuchan gritos, risas, acentos cerrados, disparos y explosiones de fondo."
)
DEFAULT_OUTPUT_PATH = Path("data/whisper/dataset_whisper.json")
WHISPER_SAMPLE_RATE = 16_000
NOISE_TOKENS = {
    "",
    "[silence]",
    "(silence)",
    "silence",
    "[music]",
    "(music)",
    "music",
    "[noise]",
    "(noise)",
    "noise",
    "[inaudible]",
    "(inaudible)",
    "inaudible",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label loud gameplay windows with Whisper transcription."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/proxy_720p30.mp4"),
        help="Input gameplay video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON dataset path.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Sliding window length in seconds.",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=DEFAULT_STRIDE_SECONDS,
        help="Sliding window stride in seconds.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Keep windows with RMS >= this value. If omitted, keep top --keep-top-pct loudest.",
    )
    parser.add_argument(
        "--keep-top-pct",
        type=float,
        default=DEFAULT_KEEP_TOP_PCT,
        help="Fraction of loudest windows to keep when --threshold is omitted.",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="Hugging Face Whisper model id.",
    )
    parser.add_argument(
        "--whisper-batch-size",
        type=int,
        default=DEFAULT_WHISPER_BATCH_SIZE,
        help="Batch size for Whisper pipeline inference.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="Initial prompt injected into Whisper generation.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Discard transcripts shorter than this many words.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Optional Whisper language, e.g. english or spanish.",
    )
    return parser.parse_args()


def slice_audio_window(audio_track: AudioTrack, window: WindowSpec) -> torch.Tensor:
    """Return one mono audio chunk for a time window at the track sample rate."""
    if not audio_track.has_audio or audio_track.waveform is None or audio_track.sample_rate is None:
        return torch.empty(1, 0, dtype=torch.float32)

    waveform = audio_track.waveform
    sample_rate = audio_track.sample_rate
    start_sample = int(round(window.start_s * sample_rate))
    end_sample = int(round(window.end_s * sample_rate))
    start_sample = max(0, min(start_sample, waveform.shape[-1]))
    end_sample = max(start_sample, min(end_sample, waveform.shape[-1]))
    return waveform[:, start_sample:end_sample].float()


def rms_energy(chunk: torch.Tensor) -> float:
    """Compute RMS energy with native PyTorch only."""
    if chunk.numel() == 0 or chunk.shape[-1] == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(chunk.pow(2))).item())


def score_windows_by_volume(
    windows: Sequence[WindowSpec],
    audio_track: AudioTrack,
) -> list[tuple[int, WindowSpec, float]]:
    scored_windows: list[tuple[int, WindowSpec, float]] = []
    for window_index, window in enumerate(windows):
        chunk = slice_audio_window(audio_track, window)
        scored_windows.append((window_index, window, rms_energy(chunk)))
    return scored_windows


def keep_loud_windows(
    scored_windows: Sequence[tuple[int, WindowSpec, float]],
    threshold: float | None,
    keep_top_pct: float,
) -> list[tuple[int, WindowSpec, float]]:
    """Keep windows over a fixed RMS threshold or the loudest percentage."""
    if threshold is not None:
        return [item for item in scored_windows if item[2] >= threshold]

    if not 0.0 < keep_top_pct <= 1.0:
        raise ValueError("--keep-top-pct must be in the range (0, 1].")

    keep_count = max(1, int(round(len(scored_windows) * keep_top_pct)))
    loudest = sorted(scored_windows, key=lambda item: item[2], reverse=True)[:keep_count]
    return sorted(loudest, key=lambda item: item[0])


def resample_for_whisper(chunk: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Whisper expects mono 16 kHz audio."""
    if chunk.numel() == 0 or chunk.shape[-1] == 0:
        return torch.empty(0, dtype=torch.float32)

    if chunk.shape[0] > 1:
        chunk = chunk.mean(dim=0, keepdim=True)
    if sample_rate != WHISPER_SAMPLE_RATE:
        chunk = torchaudio.functional.resample(
            chunk,
            orig_freq=sample_rate,
            new_freq=WHISPER_SAMPLE_RATE,
        )
    return chunk.squeeze(0).contiguous()


def load_whisper_pipeline(model_id: str, device: torch.device):
    """Load Whisper with Transformers and place it on CUDA."""
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    whisper_pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device.type == "cuda" else -1,
    )
    return whisper_pipe, processor


def build_generate_kwargs(
    processor: object,
    language: str | None,
    initial_prompt: str,
    device: torch.device,
) -> dict:
    """Build Whisper generation kwargs, including the HF prompt_ids form."""
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language
    if initial_prompt:
        prompt_ids = processor.get_prompt_ids(initial_prompt, return_tensors="pt")
        if torch.is_tensor(prompt_ids):
            prompt_ids = prompt_ids.to(device)
        generate_kwargs["prompt_ids"] = prompt_ids
    return generate_kwargs


def normalize_transcript(text: str) -> str:
    return " ".join(text.strip().split())


def is_useful_transcript(text: str, min_words: int) -> bool:
    normalized = normalize_transcript(text)
    lowered = normalized.lower()
    if lowered in NOISE_TOKENS:
        return False
    return len(normalized.split()) >= min_words


def audio_generator(
    loud_windows: Sequence[tuple[int, WindowSpec, float]],
    audio_track: AudioTrack,
):
    """Yield Whisper-ready audio dictionaries for batched pipeline inference."""
    if audio_track.sample_rate is None:
        return

    for _window_index, window, _volume in loud_windows:
        chunk = slice_audio_window(audio_track, window)
        whisper_audio = resample_for_whisper(chunk, audio_track.sample_rate)
        if whisper_audio.numel() == 0:
            whisper_audio = torch.zeros(WHISPER_SAMPLE_RATE, dtype=torch.float32)
        yield {
            "array": whisper_audio.cpu().numpy(),
            "sampling_rate": WHISPER_SAMPLE_RATE,
        }


def transcribe_loud_windows(
    loud_windows: Sequence[tuple[int, WindowSpec, float]],
    audio_track: AudioTrack,
    whisper_pipe,
    processor: object,
    min_words: int,
    language: str | None,
    initial_prompt: str,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, int | float | str]]:
    if audio_track.waveform is None or audio_track.sample_rate is None:
        return []
    if batch_size < 1:
        raise ValueError("--whisper-batch-size must be >= 1")

    labels: list[dict[str, int | float | str]] = []
    generate_kwargs = build_generate_kwargs(
        processor=processor,
        language=language,
        initial_prompt=initial_prompt,
        device=device,
    )
    results = whisper_pipe(
        audio_generator(loud_windows, audio_track),
        batch_size=batch_size,
        generate_kwargs=generate_kwargs,
    )

    for (window_index, window, _volume), result in tqdm(
        zip(loud_windows, results),
        total=len(loud_windows),
        desc="Whisper labeling",
    ):
        transcript = normalize_transcript(str(result.get("text", "")))
        if not is_useful_transcript(transcript, min_words=min_words):
            continue

        labels.append(
            {
                "window_index": int(window_index),
                "start_s": float(window.start_s),
                "end_s": float(window.end_s),
                "text_query": transcript,
            }
        )
    return labels


def save_dataset(path: Path, labels: Sequence[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(labels), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    device = require_cuda()

    _video_reader, fps, duration_s = open_video_reader(args.video)
    audio_track = load_audio_track(args.video)
    windows = make_windows(
        duration_s=duration_s,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    if not windows:
        raise RuntimeError("No windows were produced from the input video.")
    if not audio_track.has_audio:
        raise RuntimeError("No readable audio track found; Whisper auto-labeling needs audio.")

    print(
        f"Loaded video={args.video}, duration={duration_s:.2f}s, fps={fps:.2f}, "
        f"windows={len(windows)}, audio_sr={audio_track.sample_rate}"
    )

    scored_windows = score_windows_by_volume(windows, audio_track)
    loud_windows = keep_loud_windows(
        scored_windows=scored_windows,
        threshold=args.threshold,
        keep_top_pct=args.keep_top_pct,
    )
    threshold_label = (
        f"threshold >= {args.threshold:.6f}"
        if args.threshold is not None
        else f"top {args.keep_top_pct:.0%}"
    )
    print(f"Phase 1 kept {len(loud_windows)}/{len(windows)} windows ({threshold_label}).")

    whisper_pipe, processor = load_whisper_pipeline(args.whisper_model, device=device)
    labels = transcribe_loud_windows(
        loud_windows=loud_windows,
        audio_track=audio_track,
        whisper_pipe=whisper_pipe,
        processor=processor,
        min_words=args.min_words,
        language=args.language,
        initial_prompt=args.initial_prompt,
        batch_size=args.whisper_batch_size,
        device=device,
    )
    save_dataset(args.output, labels)
    print(f"Saved {len(labels)} labels to {args.output}")


if __name__ == "__main__":
    main()
