#!/usr/bin/env python3
"""Validate the local environment for the reproducible LanguageBind workflows."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_ARTIFACTS = {
    "video proxy": REPO_ROOT / "data" / "proxy_720p30.mp4",
    "precomputed index metadata": REPO_ROOT
    / "indexes"
    / "proxy_720p30_w15_s5"
    / "metadata.json",
    "precomputed video embeddings": REPO_ROOT
    / "indexes"
    / "proxy_720p30_w15_s5"
    / "video_embeddings.pt",
    "precomputed audio embeddings": REPO_ROOT
    / "indexes"
    / "proxy_720p30_w15_s5"
    / "audio_embeddings.pt",
    "precomputed scene embeddings": REPO_ROOT
    / "indexes"
    / "proxy_720p30_w15_s5"
    / "scene_embeddings.pt",
    "manual ground truth": REPO_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_dataset.json",
    "trained adapter": REPO_ROOT / "data" / "models" / "helldivers_adapter.pth",
}
FULL_ARTIFACT_DIRS = {
    "whisper data dir": REPO_ROOT / "data" / "whisper",
    "ocr data dir": REPO_ROOT / "data" / "ocr",
}

PRETRAINED_IMPORTS = (
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "decord",
    "numpy",
    "tqdm",
)
FULL_IMPORTS = (
    "cv2",
    "easyocr",
    "ollama",
    "skimage",
    "soundfile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check dependencies and artifacts for the LanguageBind project."
    )
    parser.add_argument(
        "--mode",
        choices=("pretrained", "full"),
        default="pretrained",
        help="pretrained checks only evaluation artifacts; full also checks dataset-generation dependencies.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail if CUDA is unavailable. This is useful only for metadata inspection.",
    )
    return parser.parse_args()


def print_check(ok: bool, label: str, detail: str = "") -> None:
    status = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_imports(module_names: tuple[str, ...]) -> bool:
    all_ok = True
    for module_name in module_names:
        ok = module_exists(module_name)
        print_check(ok, f"python import: {module_name}")
        all_ok = all_ok and ok
    return all_ok


def check_paths(paths: dict[str, Path]) -> bool:
    all_ok = True
    for label, path in paths.items():
        ok = path.exists()
        print_check(ok, label, str(path.relative_to(REPO_ROOT)) if ok else str(path))
        all_ok = all_ok and ok
    return all_ok


def check_cuda(allow_cpu: bool) -> bool:
    if not module_exists("torch"):
        print_check(False, "torch CUDA check", "torch is not importable")
        return False

    import torch

    print_check(True, "torch version", torch.__version__)
    cuda_available = torch.cuda.is_available()
    detail = torch.cuda.get_device_name(0) if cuda_available else "CUDA unavailable"
    print_check(cuda_available or allow_cpu, "CUDA device", detail)
    if cuda_available:
        print_check(True, "torch CUDA runtime", str(torch.version.cuda))
    return cuda_available or allow_cpu


def check_system_tools(mode: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    ok = ffmpeg is not None
    print_check(ok, "system tool: ffmpeg", ffmpeg or "required for clip export")
    if mode == "full":
        return ok
    return True


def main() -> None:
    args = parse_args()

    print(f"Repository: {REPO_ROOT}")
    print(f"Mode: {args.mode}")
    print()

    ok = True
    ok = check_imports(PRETRAINED_IMPORTS) and ok
    ok = check_cuda(allow_cpu=args.allow_cpu) and ok
    ok = check_paths(PRETRAINED_ARTIFACTS) and ok
    ok = check_system_tools(args.mode) and ok

    if args.mode == "full":
        print()
        ok = check_imports(FULL_IMPORTS) and ok
        ok = check_paths(FULL_ARTIFACT_DIRS) and ok

    print()
    if ok:
        print("Environment check passed.")
        return

    print("Environment check failed. Install missing dependencies or artifacts first.")
    sys.exit(1)


if __name__ == "__main__":
    main()
