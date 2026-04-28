#!/usr/bin/env python3
"""Evaluate the fine-tuned LanguageBind model using Multi-Target Recall@K and MRR."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_languagebind_index import load_languagebind_index, query_languagebind_index
from scripts.tfvtg_poc import load_languagebind_models, require_cuda
from scripts.train_linear_probe import VideoTextAdapter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LanguageBind Model.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth/ground_truth_dataset.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/master_dataset_final.json"))
    parser.add_argument("--adapter", type=Path, default=Path("helldivers_adapter.pth"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--video-weight", type=float, default=0.8)
    parser.add_argument("--audio-weight", type=float, default=0.2)
    return parser.parse_args()

def load_ground_truth(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_auto_ground_truth(dataset_path: Path):
    dataset = load_ground_truth(dataset_path)
    gt = []
    for item in dataset:
        gt.append({
            "query": item["text_query"],
            "target_windows": [item["window_index"]]
        })
    return gt

def calculate_metrics(results, target_windows, top_k_max=5):
    retrieved_indices = [res.window_index for res in results]
    
    mrr = 0.0
    for rank, window_idx in enumerate(retrieved_indices, start=1):
        if window_idx in target_windows:
            mrr = 1.0 / rank
            break
            
    recalls = {}
    for k in [1, 3, 5]:
        if k <= top_k_max:
            match_in_k = any(idx in target_windows for idx in retrieved_indices[:k])
            recalls[f"R@{k}"] = 1.0 if match_in_k else 0.0
            
    return mrr, recalls

def main():
    args = parse_args()
    device = require_cuda()
    print(f"Loading Index from {args.index_dir}...")
    index = load_languagebind_index(args.index_dir)
    
    if args.mode == "auto":
        print("Mode: AUTO Sanity Check (1 Target per Query)")
        queries_gt = build_auto_ground_truth(args.dataset)
    else:
        print(f"Mode: MANUAL Ground Truth (Multi-Target from {args.ground_truth})")
        queries_gt = load_ground_truth(args.ground_truth)

    print("Loading Models...")
    video_model, audio_model, _, _, text_tokenizer = load_languagebind_models(
        cache_dir=Path("cache_dir"), device=device
    )
    
    video_embeddings = index.video_embeddings.to(device=device, dtype=torch.float32)
    audio_embeddings = index.audio_embeddings.to(device=device, dtype=torch.float32)
    
    if args.adapter.exists():
        print(f"Applying Adapter from {args.adapter}...")
        embedding_dim = int(video_embeddings.shape[-1])
        adapter = VideoTextAdapter(embedding_dim=embedding_dim).to(device)
        adapter.load_state_dict(torch.load(args.adapter, map_location=device))
        adapter.eval()
        with torch.no_grad():
            video_embeddings = F.normalize(adapter(video_embeddings), dim=-1)
    else:
        print("WARNING: No adapter found. Using BASE generic embeddings.")

    total_weight = args.video_weight + args.audio_weight
    scene_embeddings = F.normalize(
        video_embeddings * (args.video_weight/total_weight) + 
        audio_embeddings * (args.audio_weight/total_weight), 
        dim=-1
    )
    
    index = replace(index, scene_embeddings=scene_embeddings)

    total_mrr = 0.0
    total_recalls = {"R@1": 0.0, "R@3": 0.0, "R@5": 0.0}
    
    print(f"\nEvaluating {len(queries_gt)} Queries...")
    with torch.no_grad():
        for item in tqdm(queries_gt):
            query = item["query"]
            targets = item["target_windows"]
            
            if not targets:
                continue # Skip empty targets
                
            results = query_languagebind_index(
                index=index,
                text_model=video_model,
                text_tokenizer=text_tokenizer,
                query=query,
                device=device,
                top_k=args.top_k,
                modality="stored_scene"
            )
            
            mrr, recalls = calculate_metrics(results, targets, args.top_k)
            total_mrr += mrr
            for k, v in recalls.items():
                total_recalls[k] += v

    valid_queries = sum(1 for q in queries_gt if q["target_windows"])
    print("\n" + "="*30)
    print("      FINAL RESULTS")
    print("="*30)
    print(f"Total Valid Queries: {valid_queries}")
    if valid_queries > 0:
        print(f"Mean Reciprocal Rank (MRR): {total_mrr / valid_queries:.4f}")
        for k in ["R@1", "R@3", "R@5"]:
            if k in total_recalls:
                print(f"Recall{k}: {total_recalls[k] / valid_queries:.2%}")
    print("="*30)

if __name__ == "__main__":
    main()
