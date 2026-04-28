#!/usr/bin/env python3
"""Train a small linear probe on top of frozen LanguageBind video embeddings."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGEBIND_CHECKOUT = REPO_ROOT / "third_party" / "LanguageBind"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if LANGUAGEBIND_CHECKOUT.exists():
    sys.path.insert(0, str(LANGUAGEBIND_CHECKOUT))

# pytorchvideo 0.1.5 imports an old torchvision module name.
try:
    import torchvision.transforms.functional as torchvision_functional

    sys.modules.setdefault(
        "torchvision.transforms.functional_tensor", torchvision_functional
    )
except Exception:
    pass

from scripts.tfvtg_poc import (  # noqa: E402
    VIDEO_MODEL_ID,
    WindowSpec,
    force_eager_attention,
    open_video_reader,
)


DEFAULT_DATASET_PATH = Path("data/master_dataset_final.json")
DEFAULT_VIDEO_PATH = Path("data/proxy_720p30.mp4")
DEFAULT_OUTPUT_PATH = Path("data/models/helldivers_adapter.pth")


class VideoTextAdapter(nn.Module):
    """Small trainable projection on top of frozen video embeddings."""

    def __init__(self, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, video_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(video_embeddings)


class HelldiversVideoTextDataset(Dataset):
    """Decode sparse video windows and tokenize their paired text descriptions."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        video_path: Path,
        video_transform: object,
        tokenizer: object,
        num_frames: int,
        context_length: int = 77,
    ) -> None:
        self.rows = list(rows)
        self.video_path = Path(video_path)
        self.video_transform = video_transform
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.context_length = context_length
        self._video_reader: object | None = None
        self._fps: float | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _reader_and_fps(self) -> tuple[object, float]:
        if self._video_reader is None or self._fps is None:
            self._video_reader, self._fps, _duration_s = open_video_reader(self.video_path)
        return self._video_reader, self._fps

    def _sample_video(self, window: WindowSpec) -> torch.Tensor:
        video_reader, fps = self._reader_and_fps()
        max_frame_index = len(video_reader) - 1
        frame_times = np.linspace(window.start_s, window.end_s, self.num_frames, endpoint=False)
        frame_indices = np.clip(np.round(frame_times * fps), 0, max_frame_index).astype("int64")
        frames = video_reader.get_batch(frame_indices).permute(3, 0, 1, 2)
        return self.video_transform(frames).contiguous()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        window = WindowSpec(start_s=float(row["start_s"]), end_s=float(row["end_s"]))
        pixel_values = self._sample_video(window)
        text = str(row["text_query"])
        tokenized = self.tokenizer(
            [text],
            max_length=self.context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear probe on frozen LanguageBind video embeddings."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache_dir"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    if not payload:
        raise ValueError(f"Dataset is empty: {path}")
    return payload


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LanguageBind linear probing.")
    return torch.device("cuda")


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def load_video_model_and_processor(cache_dir: Path, device: torch.device):
    from languagebind import LanguageBindVideo, LanguageBindVideoProcessor, LanguageBindVideoTokenizer

    model = LanguageBindVideo.from_pretrained(VIDEO_MODEL_ID, cache_dir=str(cache_dir)).to(device)
    force_eager_attention(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    tokenizer = LanguageBindVideoTokenizer.from_pretrained(VIDEO_MODEL_ID, cache_dir=str(cache_dir))
    processor = LanguageBindVideoProcessor(model.config, tokenizer)
    return model, processor, tokenizer


def embedding_dim_from_model(model: nn.Module) -> int:
    projection = getattr(model, "visual_projection", None)
    if projection is not None and hasattr(projection, "out_features"):
        return int(projection.out_features)
    projection_dim = getattr(model.config, "projection_dim", None)
    if projection_dim is None:
        raise RuntimeError("Could not infer LanguageBind embedding dimension.")
    return int(projection_dim)


def symmetric_contrastive_loss(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    video_embeddings = F.normalize(video_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    logits = video_embeddings @ text_embeddings.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_v2t = F.cross_entropy(logits, targets)
    loss_t2v = F.cross_entropy(logits.T, targets)
    return (loss_v2t + loss_t2v) / 2.0


def train() -> None:
    args = parse_args()

    device = require_cuda()
    rows = load_dataset(args.dataset)
    model, processor, tokenizer = load_video_model_and_processor(args.cache_dir, device)
    embedding_dim = embedding_dim_from_model(model)

    num_frames = int(processor.config.vision_config.num_frames)
    dataset = HelldiversVideoTextDataset(
        rows=rows,
        video_path=args.video,
        video_transform=processor.transform,
        tokenizer=tokenizer,
        num_frames=num_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    video_embedding_batches: list[torch.Tensor] = []
    text_embedding_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Precomputing frozen embeddings"):
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                video_features = model.get_image_features(pixel_values=batch["pixel_values"])
                text_features = model.get_text_features(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            video_embedding_batches.append(F.normalize(video_features.detach().float().cpu(), dim=-1))
            text_embedding_batches.append(F.normalize(text_features.detach().float().cpu(), dim=-1))
            del batch, video_features, text_features
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    video_embeddings = torch.cat(video_embedding_batches, dim=0)
    text_embeddings = torch.cat(text_embedding_batches, dim=0)
    embedding_dataset = TensorDataset(video_embeddings, text_embeddings)
    
    # FIX: Usar todo el dataset de golpe (Full Batch) para maximizar 
    # los ejemplos negativos en la matriz de Contrastive Loss.
    train_batch_size = len(embedding_dataset)
    embedding_loader = DataLoader(
        embedding_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        # Eliminamos el drop_last porque el lote es el dataset entero
    )

    adapter = VideoTextAdapter(embedding_dim=embedding_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        progress = tqdm(embedding_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (video_features, text_features) in enumerate(progress, start=1):
            video_features = video_features.to(device, non_blocking=True)
            text_features = text_features.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                adapted_video_features = adapter(video_features)
                loss = symmetric_contrastive_loss(
                    adapted_video_features,
                    text_features,
                    temperature=args.temperature,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=f"{loss.item():.4f}")

            del video_features, text_features, adapted_video_features, loss

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), args.output)
    print(f"Saved adapter weights to {args.output}")


if __name__ == "__main__":
    train()
