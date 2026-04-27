#!/usr/bin/env python3
"""Batched in-memory zero-shot video moment retrieval with LanguageBind.

This PoC avoids MoviePy and all temporary media files. Video frames are decoded
with decord directly into RAM, audio is loaded with torchaudio directly into RAM,
and sliding windows are transformed into batched tensors for high GPU utilization.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


VIDEO_MODEL_ID = "LanguageBind/LanguageBind_Video_FT"
AUDIO_MODEL_ID = "LanguageBind/LanguageBind_Audio_FT"
DEFAULT_QUERY = "sniper shot"

# Lower this first if the Blackwell GPU hits CUDA OOM on long videos.
DEFAULT_BATCH_SIZE = 48
DEFAULT_NUM_WORKERS = 4
DEFAULT_VIDEO_WEIGHT = 0.5
DEFAULT_AUDIO_WEIGHT = 0.5

# Balanced gameplay retrieval defaults: enough context for setup/impact/reaction,
# with dense overlap so short Helldivers 2 events are not skipped.
DEFAULT_WINDOW_SECONDS = 8.0
DEFAULT_STRIDE_SECONDS = 2.0
DEFAULT_CLIP_CONTEXT_SECONDS = 2.0

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGEBIND_CHECKOUT = REPO_ROOT / "third_party" / "LanguageBind"

if LANGUAGEBIND_CHECKOUT.exists():
    sys.path.insert(0, str(LANGUAGEBIND_CHECKOUT))

# pytorchvideo 0.1.5 imports an old torchvision module name. Current torchvision
# exposes the same tensor transform functions through torchvision.transforms.functional.
try:
    import torchvision.transforms.functional as torchvision_functional

    sys.modules.setdefault(
        "torchvision.transforms.functional_tensor", torchvision_functional
    )
except Exception:
    pass


@dataclass(frozen=True)
class WindowSpec:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    window_index: int
    start_s: float
    end_s: float
    score: float
    clip_path: Path | None = None


@dataclass(frozen=True)
class AudioTrack:
    waveform: torch.Tensor | None  # [1, total_samples] float32 on CPU, or None.
    sample_rate: int | None
    has_audio: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched LanguageBind zero-shot video moment retrieval PoC."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/proxy_720p30.mp4"),
        help="Input .mp4 path.",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker processes for async video/audio preprocessing.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--video-weight", type=float, default=DEFAULT_VIDEO_WEIGHT)
    parser.add_argument("--audio-weight", type=float, default=DEFAULT_AUDIO_WEIGHT)
    parser.add_argument(
        "--export-clips",
        action="store_true",
        help="Export retrieved top-k windows as validation MP4 clips.",
    )
    parser.add_argument(
        "--clip-output-dir",
        type=Path,
        default=Path("outputs/validation_clips"),
        help="Directory for exported validation clips when --export-clips is set.",
    )
    parser.add_argument(
        "--clip-context-seconds",
        type=float,
        default=DEFAULT_CLIP_CONTEXT_SECONDS,
        help="Extra seconds before/after retrieved windows in exported clips.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache_dir"),
        help="Hugging Face/model cache directory.",
    )
    return parser.parse_args()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this PoC; no CUDA device was found.")
    return torch.device("cuda")


def move_to_device(batch: object, device: torch.device) -> object:
    """Recursively move tensors or Hugging Face BatchEncoding objects to CUDA."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def force_eager_attention(model: torch.nn.Module) -> None:
    """Patch older LanguageBind configs for modern Transformers attention dispatch.

    Transformers 4.57's CLIPAttention indexes ALL_ATTENTION_FUNCTIONS using
    config._attn_implementation. LanguageBind's older CLIP-derived configs may
    leave that field as None, which raises KeyError during text/video/audio
    forward passes. Eager attention is the most compatible implementation.
    """
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None:
            setattr(config, "_attn_implementation", "eager")
            vision_config = getattr(config, "vision_config", None)
            if vision_config is not None:
                setattr(vision_config, "_attn_implementation", "eager")
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                setattr(text_config, "_attn_implementation", "eager")


def patch_rectangular_clip_embeddings(model: torch.nn.Module) -> None:
    """Allow CLIPVisionEmbeddings to accept rectangular spectrogram inputs.

    LanguageBind Audio resizes its CLIP vision embedding table to rectangular
    spectrogram geometry: [num_mel_bins, target_length], usually [112, 1036].
    Modern Transformers' CLIPVisionEmbeddings.forward assumes image_size is a
    scalar and compares height/width against the whole list, causing a false
    ValueError for correctly shaped audio tensors. This instance-level patch
    keeps the same computation while validating rectangular sizes properly.
    """
    embeddings = getattr(getattr(model, "vision_model", None), "embeddings", None)
    if embeddings is None or not isinstance(getattr(embeddings, "image_size", None), (list, tuple)):
        return

    def rectangular_forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False):
        batch_size, _, height, width = pixel_values.shape
        expected_height, expected_width = int(self.image_size[0]), int(self.image_size[1])
        if not interpolate_pos_encoding and (height != expected_height or width != expected_width):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model "
                f"({expected_height}*{expected_width})."
            )

        # pixel_values shape: [B, 3, 112, 1036] for LanguageBind Audio.
        # patch_embeds shape after Conv2d/flatten: [B, num_patches, hidden_dim].
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        output = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            output = output + self.interpolate_pos_encoding(output, height, width)
        else:
            output = output + self.position_embedding(self.position_ids.to(pixel_values.device))
        return output

    embeddings.forward = types.MethodType(rectangular_forward, embeddings)


def weighted_fuse_embeddings(
    video_embedding: torch.Tensor,
    audio_embedding: torch.Tensor,
    video_weight: float = DEFAULT_VIDEO_WEIGHT,
    audio_weight: float = DEFAULT_AUDIO_WEIGHT,
) -> torch.Tensor:
    """Fuse normalized video/audio embeddings with configurable modality weights."""
    total_weight = video_weight + audio_weight
    if total_weight <= 0:
        raise ValueError("video_weight + audio_weight must be > 0")
    video_scale = video_weight / total_weight
    audio_scale = audio_weight / total_weight
    return F.normalize(video_embedding * video_scale + audio_embedding * audio_scale, dim=-1)


def load_languagebind_models(cache_dir: Path, device: torch.device):
    """Load LanguageBind video/audio branches and processors."""
    from languagebind import (
        LanguageBindAudio,
        LanguageBindAudioProcessor,
        LanguageBindAudioTokenizer,
        LanguageBindVideo,
        LanguageBindVideoProcessor,
        LanguageBindVideoTokenizer,
    )

    video_model = LanguageBindVideo.from_pretrained(
        VIDEO_MODEL_ID, cache_dir=str(cache_dir)
    ).to(device)
    audio_model = LanguageBindAudio.from_pretrained(
        AUDIO_MODEL_ID, cache_dir=str(cache_dir)
    ).to(device)
    force_eager_attention(video_model)
    force_eager_attention(audio_model)
    patch_rectangular_clip_embeddings(audio_model)
    video_model.eval()
    audio_model.eval()

    video_tokenizer = LanguageBindVideoTokenizer.from_pretrained(
        VIDEO_MODEL_ID, cache_dir=str(cache_dir)
    )
    audio_tokenizer = LanguageBindAudioTokenizer.from_pretrained(
        AUDIO_MODEL_ID, cache_dir=str(cache_dir)
    )
    video_processor = LanguageBindVideoProcessor(video_model.config, video_tokenizer)
    audio_processor = LanguageBindAudioProcessor(audio_model.config, audio_tokenizer)
    return video_model, audio_model, video_processor, audio_processor, video_tokenizer


def make_windows(duration_s: float, window_seconds: float, stride_seconds: float) -> list[WindowSpec]:
    """Return overlapping [start, end] windows in seconds."""
    if window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if stride_seconds <= 0:
        raise ValueError("--stride-seconds must be > 0")

    windows: list[WindowSpec] = []
    start_s = 0.0
    while start_s < duration_s:
        end_s = min(start_s + window_seconds, duration_s)
        if end_s <= start_s:
            break
        windows.append(WindowSpec(start_s=start_s, end_s=end_s))
        if end_s >= duration_s:
            break
        start_s += stride_seconds
    return windows


def open_video_reader(video_path: Path):
    """Open decord.VideoReader and return (reader, fps, duration_s)."""
    import decord
    from decord import VideoReader, cpu

    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    decord.bridge.set_bridge("torch")
    reader = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        raise RuntimeError(f"Could not determine FPS for {video_path}")
    duration_s = len(reader) / fps
    return reader, fps, duration_s


def load_audio_track(video_path: Path, target_sample_rate: int | None = None) -> AudioTrack:
    """Load raw mono audio into RAM; return a silent marker if no audio is readable.

    target_sample_rate is accepted for older callers, but resampling now happens
    per sliced window in WindowDataset.
    """
    try:
        waveform, sample_rate = torchaudio.load(str(video_path))
    except Exception as exc:
        print(f"Warning: could not load audio track ({exc}); using silence.")
        return AudioTrack(waveform=None, sample_rate=None, has_audio=False)

    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        print("Warning: empty audio track; using silence.")
        return AudioTrack(waveform=None, sample_rate=None, has_audio=False)

    # LanguageBind's audio transform expects [channels, samples]. We mix to mono
    # so every window has shape [1, samples] before mel extraction.
    waveform = waveform.float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return AudioTrack(waveform=waveform.contiguous(), sample_rate=sample_rate, has_audio=True)


def sample_window_frames(
    video_reader: object,
    fps: float,
    window: WindowSpec,
    num_frames: int,
    video_transform: object,
) -> torch.Tensor:
    """Decode one window to LanguageBind video tensor [C, T, H, W] on CPU.

    decord returns frames as [T, H, W, C] uint8 RGB. LanguageBind's transform
    expects [C, T, H, W], normalizes to float, scales/crops to 224, and keeps
    the temporal dimension T equal to the model config's num_frames.
    """
    max_frame_index = len(video_reader) - 1
    frame_times = np.linspace(window.start_s, window.end_s, num_frames, endpoint=False)
    frame_indices = np.clip(np.round(frame_times * fps), 0, max_frame_index).astype("int64")
    frames = video_reader.get_batch(frame_indices).permute(3, 0, 1, 2)
    return video_transform(frames)


def sample_window_audio(
    audio_track: AudioTrack,
    window: WindowSpec,
    target_sample_rate: int,
    audio_transform: object,
) -> torch.Tensor:
    """Slice one audio window and transform to LanguageBind tensor [3, M, T] on CPU.

    The audio transform resamples if needed, computes log-mel filterbanks, pads
    or chunks to the model target length, and outputs [3, num_mel_bins, target_len].
    Missing or too-short audio is represented as silence so video-only assets do
    not crash the retrieval run.
    """
    if not audio_track.has_audio or audio_track.waveform is None or audio_track.sample_rate is None:
        num_samples = max(1, int(math.ceil((window.end_s - window.start_s) * target_sample_rate)))
        waveform = torch.zeros(1, num_samples, dtype=torch.float32)
        return audio_transform((waveform, target_sample_rate))

    waveform = audio_track.waveform
    sample_rate = audio_track.sample_rate
    start_sample = int(round(window.start_s * sample_rate))
    end_sample = int(round(window.end_s * sample_rate))
    start_sample = max(0, min(start_sample, waveform.shape[-1]))
    end_sample = max(start_sample, min(end_sample, waveform.shape[-1]))

    chunk = waveform[:, start_sample:end_sample]
    if chunk.numel() == 0 or chunk.shape[-1] == 0:
        num_samples = max(1, int(math.ceil((window.end_s - window.start_s) * sample_rate)))
        chunk = torch.zeros(1, num_samples, dtype=torch.float32)
    else:
        chunk = chunk.clone()

    if sample_rate != target_sample_rate:
        chunk = torchaudio.functional.resample(
            chunk,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        sample_rate = target_sample_rate
    return audio_transform((chunk, sample_rate))


class WindowDataset(Dataset):
    """Decode and transform one video/audio window per item.

    decord.VideoReader is intentionally opened lazily inside each worker process:
    it owns a C++ reader that cannot be pickled or shared safely across workers.
    """

    def __init__(
        self,
        windows: Sequence[WindowSpec],
        video_path: Path,
        raw_waveform: torch.Tensor | None,
        raw_sample_rate: int | None,
        num_frames: int,
        target_audio_sample_rate: int,
        video_transform: object,
        audio_transform: object,
    ) -> None:
        self.windows = list(windows)
        self.video_path = Path(video_path)
        self.audio_track = AudioTrack(
            waveform=raw_waveform,
            sample_rate=raw_sample_rate,
            has_audio=raw_waveform is not None and raw_sample_rate is not None,
        )
        self.num_frames = num_frames
        self.target_audio_sample_rate = target_audio_sample_rate
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self._video_reader: object | None = None
        self._fps: float | None = None

    def __len__(self) -> int:
        return len(self.windows)

    def _reader_and_fps(self) -> tuple[object, float]:
        if self._video_reader is None or self._fps is None:
            self._video_reader, self._fps, _duration_s = open_video_reader(self.video_path)
        return self._video_reader, self._fps

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        window = self.windows[index]
        video_reader, fps = self._reader_and_fps()
        video_tensor = sample_window_frames(
            video_reader=video_reader,
            fps=fps,
            window=window,
            num_frames=self.num_frames,
            video_transform=self.video_transform,
        )
        audio_tensor = sample_window_audio(
            audio_track=self.audio_track,
            window=window,
            target_sample_rate=self.target_audio_sample_rate,
            audio_transform=self.audio_transform,
        )
        return (
            {"pixel_values": video_tensor.contiguous()},
            {"pixel_values": audio_tensor.contiguous()},
        )


def collate_window_batch(
    batch: Sequence[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    video_tensors = [item[0]["pixel_values"] for item in batch]
    audio_tensors = [item[1]["pixel_values"] for item in batch]
    return (
        {"pixel_values": torch.stack(video_tensors, dim=0).contiguous()},
        {"pixel_values": torch.stack(audio_tensors, dim=0).contiguous()},
    )


def extract_text_embedding(
    model: torch.nn.Module,
    tokenizer: object,
    query: str,
    device: torch.device,
) -> torch.Tensor:
    batch = tokenizer(
        [query],
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    batch = move_to_device(batch, device)
    return F.normalize(model.get_text_features(**batch), dim=-1)


def extract_scene_embeddings_batched(
    windows: Sequence[WindowSpec],
    video_path: Path,
    video_reader: object,
    fps: float,
    audio_track: AudioTrack,
    video_model: torch.nn.Module,
    audio_model: torch.nn.Module,
    video_processor: object,
    audio_processor: object,
    device: torch.device,
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
    video_weight: float = DEFAULT_VIDEO_WEIGHT,
    audio_weight: float = DEFAULT_AUDIO_WEIGHT,
) -> torch.Tensor:
    """Encode all windows in GPU batches and return [num_windows, dim] scene embeddings."""
    _, _, scene_embeddings = extract_modal_embeddings_batched(
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
    return scene_embeddings


def extract_modal_embeddings_batched(
    windows: Sequence[WindowSpec],
    video_path: Path,
    video_reader: object,
    fps: float,
    audio_track: AudioTrack,
    video_model: torch.nn.Module,
    audio_model: torch.nn.Module,
    video_processor: object,
    audio_processor: object,
    device: torch.device,
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
    video_weight: float = DEFAULT_VIDEO_WEIGHT,
    audio_weight: float = DEFAULT_AUDIO_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode all windows and return video/audio/fused embeddings.

    Returned tensors stay on CUDA so callers can immediately rank or move them
    to CPU for persistence.
    """
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    # Kept in the signature because callers still use the parent reader for
    # duration/metadata; worker-local readers are opened inside WindowDataset.
    _ = (video_reader, fps)

    video_embeddings: list[torch.Tensor] = []
    audio_embeddings: list[torch.Tensor] = []
    scene_embeddings: list[torch.Tensor] = []
    num_batches = math.ceil(len(windows) / batch_size)
    num_frames = int(video_processor.config.vision_config.num_frames)
    target_audio_sr = int(audio_processor.config.vision_config.audio_sample_rate)
    dataset = WindowDataset(
        windows=windows,
        video_path=video_path,
        raw_waveform=audio_track.waveform,
        raw_sample_rate=audio_track.sample_rate,
        num_frames=num_frames,
        target_audio_sample_rate=target_audio_sr,
        video_transform=video_processor.transform,
        audio_transform=audio_processor.transform,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_window_batch,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    data_loader = DataLoader(dataset, **loader_kwargs)

    for batch_index, (video_batch, audio_batch) in enumerate(data_loader, start=1):
        batch_window_count = int(video_batch["pixel_values"].shape[0])

        video_batch = move_to_device(video_batch, device)
        audio_batch = move_to_device(audio_batch, device)

        # Batched model execution:
        # video_embedding/audio_embedding shapes are [B, D]. They are normalized
        # before fusion so neither modality dominates by vector magnitude.
        video_embedding = F.normalize(video_model.get_image_features(**video_batch), dim=-1)
        audio_embedding = F.normalize(audio_model.get_image_features(**audio_batch), dim=-1)

        # Late Fusion: LanguageBind aligns video/audio to the same text-centered
        # space, so we combine normalized modality embeddings element-wise. The
        # weights can bias retrieval toward visual gameplay cues or voice/audio.
        scene_embedding = weighted_fuse_embeddings(
            video_embedding,
            audio_embedding,
            video_weight=video_weight,
            audio_weight=audio_weight,
        )
        video_embeddings.append(video_embedding.detach())
        audio_embeddings.append(audio_embedding.detach())
        scene_embeddings.append(scene_embedding.detach())

        del video_batch, audio_batch, video_embedding, audio_embedding, scene_embedding
        if batch_index < num_batches:
            torch.cuda.empty_cache()
        print(f"Encoded batch {batch_index}/{num_batches} ({batch_window_count} windows)")

    return (
        torch.cat(video_embeddings, dim=0),
        torch.cat(audio_embeddings, dim=0),
        torch.cat(scene_embeddings, dim=0),
    )


def rank_windows(
    windows: Sequence[WindowSpec],
    scene_embeddings: torch.Tensor,
    text_embedding: torch.Tensor,
    top_k: int,
) -> list[RetrievalResult]:
    similarities = F.cosine_similarity(scene_embeddings, text_embedding, dim=-1)
    k = min(top_k, similarities.numel())
    scores, indices = torch.topk(similarities, k=k)

    results: list[RetrievalResult] = []
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        window = windows[index]
        results.append(
            RetrievalResult(
                rank=rank,
                window_index=index,
                start_s=window.start_s,
                end_s=window.end_s,
                score=score,
            )
        )
    return results


def print_results(results: Sequence[RetrievalResult]) -> None:
    print(f"\nTop {len(results)} windows:")
    for result in results:
        clip_suffix = f" clip={result.clip_path}" if result.clip_path is not None else ""
        print(
            f"{result.rank}. {result.start_s:7.2f}s - {result.end_s:7.2f}s "
            f"score={result.score:.4f}{clip_suffix}"
        )


def export_result_clips(
    video_path: Path,
    results: Sequence[RetrievalResult],
    output_dir: Path,
    query: str,
    context_seconds: float = 0.0,
    max_duration_s: float | None = None,
) -> list[RetrievalResult]:
    """Export only the retrieved windows as real MP4 clips for validation.

    This intentionally writes just the top-k validation clips, not thousands of
    sliding-window intermediates. It uses ffmpeg directly, avoiding MoviePy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("ffmpeg is required to export validation clips.") from exc

    safe_query = "".join(ch if ch.isalnum() else "_" for ch in query.lower()).strip("_")
    safe_query = safe_query or "query"
    exported: list[RetrievalResult] = []

    for result in results:
        export_start_s = max(0.0, result.start_s - context_seconds)
        export_end_s = result.end_s + context_seconds
        if max_duration_s is not None:
            export_end_s = min(max_duration_s, export_end_s)
        duration_s = max(0.001, export_end_s - export_start_s)
        clip_path = output_dir / (
            f"rank_{result.rank:02d}_{safe_query}_"
            f"hit_{result.start_s:.2f}s_{result.end_s:.2f}s_"
            f"clip_{export_start_s:.2f}s_{export_end_s:.2f}s.mp4"
        )
        command = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{export_start_s:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration_s:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(clip_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        exported.append(
            RetrievalResult(
                rank=result.rank,
                window_index=result.window_index,
                start_s=result.start_s,
                end_s=result.end_s,
                score=result.score,
                clip_path=clip_path,
            )
        )
    return exported


def print_top_windows(
    windows: Sequence[WindowSpec],
    scene_embeddings: torch.Tensor,
    text_embedding: torch.Tensor,
    top_k: int,
) -> None:
    print_results(rank_windows(windows, scene_embeddings, text_embedding, top_k))


def main() -> None:
    args = parse_args()
    device = require_cuda()

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    (
        video_model,
        audio_model,
        video_processor,
        audio_processor,
        text_tokenizer,
    ) = load_languagebind_models(cache_dir=cache_dir, device=device)

    video_reader, fps, duration_s = open_video_reader(args.video)
    audio_track = load_audio_track(args.video)
    windows = make_windows(
        duration_s=duration_s,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )
    if not windows:
        raise RuntimeError("No windows were produced from the input video.")

    print(
        f"Loaded {args.video} in memory: duration={duration_s:.2f}s, "
        f"fps={fps:.2f}, windows={len(windows)}, batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, "
        f"audio={'yes' if audio_track.has_audio else 'no/silence'}"
    )

    # The entire model forward path is inside no_grad(). Processor outputs are
    # moved to CUDA before every batched model call.
    with torch.no_grad():
        text_embedding = extract_text_embedding(
            video_model,
            text_tokenizer,
            args.query,
            device,
        )
        scene_embeddings = extract_scene_embeddings_batched(
            windows=windows,
            video_path=args.video,
            video_reader=video_reader,
            fps=fps,
            audio_track=audio_track,
            video_model=video_model,
            audio_model=audio_model,
            video_processor=video_processor,
            audio_processor=audio_processor,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            video_weight=args.video_weight,
            audio_weight=args.audio_weight,
        )
        results = rank_windows(windows, scene_embeddings, text_embedding, args.top_k)

    if args.export_clips:
        results = export_result_clips(
            video_path=args.video,
            results=results,
            output_dir=args.clip_output_dir,
            query=args.query,
            context_seconds=args.clip_context_seconds,
            max_duration_s=duration_s,
        )
    print_results(results)


if __name__ == "__main__":
    main()
