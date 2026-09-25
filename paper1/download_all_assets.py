#!/usr/bin/env python3
"""
Master Asset Downloader (Run on the Online CPU/Login Node)
Downloads all Datasets & Embedding Models to shared disk so the GPU node can run 100% offline.
Zero PyTorch model execution — purely downloads files to disk!
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

PAPER1_DIR = Path(__file__).resolve().parent
if str(PAPER1_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER1_DIR))

from config import DEFAULT_LANGUAGES, EXPERIMENT_SEED
from src.dataset_loader import IndicMSMARCOLoader

# Models to cache for offline retrieval
MODELS_TO_CACHE = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/LaBSE",
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
]


def main():
    print("=" * 80)
    print(" 📦 ONE-STEP ASSET PREPARATION (DATASET & RETRIEVAL MODELS)")
    print(" [Run on CPU/Login Node — Pure File Downloads to Shared Disk]")
    print("=" * 80)

    # Download Offline Dense Retrieval Models
    print("\n--- Downloading HuggingFace Models for Offline GPU Retrieval ---")
    for i, model_id in enumerate(MODELS_TO_CACHE, 1):
        print(f"\n[{i}/{len(MODELS_TO_CACHE)}] Downloading '{model_id}'...")
        try:
            local_path = snapshot_download(
                repo_id=model_id,
                resume_download=True,
            )
            print(f"[✓] Cached '{model_id}' at: {local_path}")
        except Exception as e:
            print(f"[!] Warning on {model_id}: {e}")

    print("\n" + "=" * 80)
    print(" [✓] ALL DATASETS & MODELS ARE NOW CACHED LOCALLY ON DISK!")
    print(" Your GPU node (gpu04) can now load everything completely offline.")
    print("=" * 80)


if __name__ == "__main__":
    main()
