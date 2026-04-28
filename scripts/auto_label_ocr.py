#!/usr/bin/env python3
"""Auto-label Helldivers 2 gameplay windows from HUD OCR keywords."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tfvtg_poc import WindowSpec, make_windows, open_video_reader  # noqa: E402


DEFAULT_OUTPUT_PATH = Path("data/ocr/dataset_ocr_raw.json")
DEFAULT_WINDOW_SECONDS = 15.0
DEFAULT_STRIDE_SECONDS = 5.0
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_OCR_BATCH_SIZE = 32
DEFAULT_FRAME_BATCH_SIZE = 64
ROI_PRESETS = ("full", "hud")
CENTER_BANNER_EVENTS = {
    "ELIMINADO": "ELIMINADO",
    "BRECHA": "BRECHA DE BICHOS",
    "NO DISPONIBLE": "REFUERZO NO DISPONIBLE",
    "REFUERZOS LISTOS": "REFUERZOS LISTOS",
    "COMPLETADA": "MISIÓN COMPLETADA",
    "COMPLETADO": "MISIÓN COMPLETADA",
}
STRATAGEM_MENU_KEYWORDS = (
    "ORBITAL",
    "ÁGUILA",
    "BOMBA",
    "REABASTECIMIENTO",
    "BALIZA",
    "ATAQUE",
    "RECUPERACIÓN",
)
SQUAD_DOWN_KEYWORDS = ("DISPONIBLE", "DISPONIBLES")
SHIP_MENU_KEYWORDS = (
    "GESTIÓN",
    "NAVE",
    "ADQUISICIONES",
    "ARMERÍA",
    "DESTRUCTOR",
)
EVENT_ORDER = (
    "EN LA NAVE",
    "ELIMINADO",
    "BRECHA DE BICHOS",
    "REFUERZO NO DISPONIBLE",
    "REFUERZOS LISTOS",
    "MISIÓN COMPLETADA",
    "USO DE ESTRATAGEMA",
    "COMPAÑERO CAÍDO",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label gameplay windows by OCRing Helldivers 2 HUD keywords."
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
        help="Output OCR dataset JSON path.",
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
        "--roi",
        choices=ROI_PRESETS,
        default="hud",
        help="OCR region preset. 'hud' reads strict center/top-left/bottom-left HUD areas.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum EasyOCR confidence to consider a text hit.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Uppercase and strip accents so EXTRACCION matches EXTRACCIÓN."""
    normalized = unicodedata.normalize("NFKD", text.upper())
    return "".join(char for char in normalized if not unicodedata.combining(char))


NORMALIZED_CENTER_BANNER_EVENTS = {
    normalize_text(keyword): event for keyword, event in CENTER_BANNER_EVENTS.items()
}
NORMALIZED_STRATAGEM_MENU_KEYWORDS = tuple(
    normalize_text(keyword) for keyword in STRATAGEM_MENU_KEYWORDS
)
NORMALIZED_SQUAD_DOWN_KEYWORDS = tuple(
    normalize_text(keyword) for keyword in SQUAD_DOWN_KEYWORDS
)
NORMALIZED_SHIP_MENU_KEYWORDS = tuple(
    normalize_text(keyword) for keyword in SHIP_MENU_KEYWORDS
)


def subsampled_frame_indices(
    window: WindowSpec,
    fps: float,
    max_frame_index: int,
    num_samples: int | None = None,
) -> np.ndarray:
    """Return one frame per second for temporal OCR/CV sampling."""
    if num_samples is None:
        duration_s = max(0.0, float(window.end_s - window.start_s))
        num_samples = max(1, int(np.ceil(duration_s)))

    frame_times = np.linspace(window.start_s, window.end_s, num_samples, endpoint=False)
    frame_indices = np.clip(np.round(frame_times * fps), 0, max_frame_index)
    return frame_indices.astype("int64")


def frame_to_numpy(frame: Any) -> np.ndarray:
    """Convert a decord torch/NDArray frame to uint8 RGB numpy."""
    if torch.is_tensor(frame):
        array = frame.cpu().numpy()
    elif hasattr(frame, "asnumpy"):
        array = frame.asnumpy()
    else:
        array = np.asarray(frame)
    return array.astype(np.uint8, copy=False)


def preprocess_for_hud_ocr(frame_rgb: np.ndarray) -> np.ndarray:
    """Boost white/red HUD text and return a high-contrast grayscale image."""
    red = frame_rgb[:, :, 0].astype(np.int16)
    green = frame_rgb[:, :, 1].astype(np.int16)
    blue = frame_rgb[:, :, 2].astype(np.int16)

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    white_mask = gray > 165
    red_mask = (red > 135) & (red > green + 35) & (red > blue + 35)
    hud_mask = np.where(white_mask | red_mask, 255, 0).astype(np.uint8)

    # Small dilation reconnects thin HUD glyphs after thresholding.
    kernel = np.ones((2, 2), dtype=np.uint8)
    hud_mask = cv2.dilate(hud_mask, kernel, iterations=1)
    return hud_mask


def roi_slices(frame_shape: tuple[int, int, int], preset: str) -> list[tuple[str, slice, slice]]:
    height, width = frame_shape[:2]

    return [
        (
            "center",
            slice(int(height * 0.25), int(height * 0.64)),
            slice(int(width * 0.20), int(width * 0.80)),
        ),
        (
            "top_header",
            slice(0, int(height * 0.15)),
            slice(0, int(width * 0.60)),
        ),
        (
            "top_left",
            slice(0, int(height * 0.42)),
            slice(0, int(width * 0.45)),
        ),
        (
            "bottom_left",
            slice(int(height * 0.58), height),
            slice(0, int(width * 0.36)),
        ),
    ]


def events_from_ocr_results(
    roi_name: str,
    results: Sequence[tuple[Any, str, float]],
    min_confidence: float,
) -> set[str]:
    detected_events: set[str] = set()

    for _bbox, text, confidence in results:
        if confidence < min_confidence:
            continue
        normalized = normalize_text(str(text))

        if roi_name == "center":
            for normalized_keyword, event in NORMALIZED_CENTER_BANNER_EVENTS.items():
                if normalized_keyword in normalized:
                    detected_events.add(event)

        if roi_name == "top_header":
            if any(keyword in normalized for keyword in NORMALIZED_SHIP_MENU_KEYWORDS):
                detected_events.add("EN LA NAVE")

        if roi_name == "top_left":
            if any(keyword in normalized for keyword in NORMALIZED_STRATAGEM_MENU_KEYWORDS):
                detected_events.add("USO DE ESTRATAGEMA")

        if roi_name == "bottom_left":
            if any(keyword in normalized for keyword in NORMALIZED_SQUAD_DOWN_KEYWORDS):
                detected_events.add("COMPAÑERO CAÍDO")

    return detected_events


def detected_keywords_from_ocr(
    reader: Any,
    processed_frame: np.ndarray,
    original_shape: tuple[int, int, int],
    roi_preset: str,
    min_confidence: float,
) -> set[str]:
    detected_events: set[str] = set()

    for roi_name, y_slice, x_slice in roi_slices(original_shape, roi_preset):
        crop = processed_frame[y_slice, x_slice]
        if crop.size == 0:
            continue

        with torch.inference_mode():
            results = reader.readtext(crop, detail=1, paragraph=False)
        detected_events.update(events_from_ocr_results(roi_name, results, min_confidence))

    return detected_events


def batch_detect_keywords_from_ocr(
    reader: Any,
    processed_frames: Sequence[np.ndarray],
    original_shapes: Sequence[tuple[int, int, int]],
    roi_preset: str,
    min_confidence: float,
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE,
) -> list[set[str]]:
    frame_events = [set() for _frame in processed_frames]
    roi_batches: dict[str, list[tuple[int, np.ndarray]]] = {}

    for frame_position, (processed_frame, original_shape) in enumerate(
        zip(processed_frames, original_shapes)
    ):
        for roi_name, y_slice, x_slice in roi_slices(original_shape, roi_preset):
            crop = processed_frame[y_slice, x_slice]
            if crop.size == 0:
                continue
            roi_batches.setdefault(roi_name, []).append((frame_position, crop))

    for roi_name, indexed_crops in roi_batches.items():
        for batch_start in range(0, len(indexed_crops), ocr_batch_size):
            batch = indexed_crops[batch_start : batch_start + ocr_batch_size]
            frame_positions = [frame_position for frame_position, _crop in batch]
            crops = [crop for _frame_position, crop in batch]

            with torch.inference_mode():
                result_batch = reader.readtext_batched(
                    crops,
                    batch_size=ocr_batch_size,
                    detail=1,
                    paragraph=False,
                )

            for frame_position, results in zip(frame_positions, result_batch):
                frame_events[frame_position].update(
                    events_from_ocr_results(roi_name, results, min_confidence)
                )

    return frame_events


def detect_events_for_frames(
    video_reader: Any,
    frame_indices: Sequence[int],
    reader: Any,
    roi_preset: str,
    min_confidence: float,
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE,
) -> dict[int, set[str]]:
    unique_frame_indices = list(dict.fromkeys(int(index) for index in frame_indices))
    if not unique_frame_indices:
        return {}

    frame_batch = frame_to_numpy(video_reader.get_batch(unique_frame_indices))
    processed_frames: list[np.ndarray] = []
    original_shapes: list[tuple[int, int, int]] = []
    frame_events: dict[int, set[str]] = {}

    for frame_index, frame_rgb in zip(unique_frame_indices, frame_batch):
        events: set[str] = set()
        processed_frames.append(preprocess_for_hud_ocr(frame_rgb))
        original_shapes.append(frame_rgb.shape)
        frame_events[frame_index] = events

    ocr_event_batches = batch_detect_keywords_from_ocr(
        reader=reader,
        processed_frames=processed_frames,
        original_shapes=original_shapes,
        roi_preset=roi_preset,
        min_confidence=min_confidence,
        ocr_batch_size=ocr_batch_size,
    )
    for frame_index, ocr_events in zip(unique_frame_indices, ocr_event_batches):
        frame_events[frame_index].update(ocr_events)

    return frame_events


def build_ocr_dataset(
    video_path: Path,
    output_path: Path,
    window_seconds: float,
    stride_seconds: float,
    roi_preset: str,
    min_confidence: float,
) -> list[dict[str, int | float | str]]:
    video_reader, fps, duration_s = open_video_reader(video_path)
    windows = make_windows(
        duration_s=duration_s,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )
    if not windows:
        raise RuntimeError("No windows were produced from the input video.")

    import easyocr

    reader = easyocr.Reader(lang_list=["es"], gpu=True)
    labels: list[dict[str, int | float | str]] = []
    frame_event_cache: dict[int, set[str]] = {}
    max_frame_index = len(video_reader) - 1
    window_frame_indices = [
        subsampled_frame_indices(
            window=window,
            fps=fps,
            max_frame_index=max_frame_index,
        )
        for window in windows
    ]
    unique_frame_indices = list(
        dict.fromkeys(
            int(frame_index)
            for frame_indices in window_frame_indices
            for frame_index in frame_indices
        )
    )

    for batch_start in tqdm(
        range(0, len(unique_frame_indices), DEFAULT_FRAME_BATCH_SIZE),
        desc="OCR frames",
    ):
        batch_indices = unique_frame_indices[
            batch_start : batch_start + DEFAULT_FRAME_BATCH_SIZE
        ]
        frame_event_cache.update(
            detect_events_for_frames(
                video_reader=video_reader,
                frame_indices=batch_indices,
                reader=reader,
                roi_preset=roi_preset,
                min_confidence=min_confidence,
            )
        )

    for window_index, (window, frame_indices) in enumerate(
        tqdm(zip(windows, window_frame_indices), total=len(windows), desc="OCR labeling")
    ):
        window_events: set[str] = set()
        for frame_index in frame_indices:
            window_events.update(frame_event_cache[int(frame_index)])

        if window_events:
            if "EN LA NAVE" in window_events:
                ordered_events = ["EN LA NAVE"]
            else:
                ordered_events = [
                    event for event in EVENT_ORDER if event in window_events
                ]
            labels.append(
                {
                    "window_index": int(window_index),
                    "start_s": float(window.start_s),
                    "end_s": float(window.end_s),
                    "text_query": ", ".join(ordered_events),
                }
            )

    save_dataset(output_path, labels)
    return labels


def save_dataset(path: Path, labels: Sequence[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(labels), handle, indent=2, ensure_ascii=False)


def release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> None:
    args = parse_args()
    try:
        labels = build_ocr_dataset(
            video_path=args.video,
            output_path=args.output,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            roi_preset=args.roi,
            min_confidence=args.min_confidence,
        )
        print(f"Saved {len(labels)} OCR labels to {args.output}")
    finally:
        release_gpu_memory()


if __name__ == "__main__":
    main()
