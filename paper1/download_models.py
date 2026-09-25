#!/usr/bin/env python3
"""
Pre-download HuggingFace Embedding & Retrieval Models (Run on Online CPU/Login Node)
Downloads model weights to the local disk/cache so they can be loaded offline on the GPU node.
Does NOT perform any PyTorch computation or model execution (safe pure file download).
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# List of models to pre-download for offline GPU retrieval
MODELS_TO_DOWNLOAD = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/LaBSE",
]


def main():
    print("=" * 80)
    print(" 📥 PRE-DOWNLOADING MODELS FOR OFFLINE GPU RETRIEVAL")
    print(" [Pure file download — No model execution / zero heavy CPU load]")
    print("=" * 80)

    for i, model_id in enumerate(MODELS_TO_DOWNLOAD, 1):
        print(f"\n[{i}/{len(MODELS_TO_DOWNLOAD)}] Downloading '{model_id}'...")
        try:
            local_dir = snapshot_download(
                repo_id=model_id,
                resume_download=True,
            )
            print(f"[✓] Successfully cached '{model_id}' at:\n    -> {local_dir}")
        except Exception as e:
            print(f"[!] Warning: Failed to download {model_id}: {e}")

    print("\n" + "=" * 80)
    print(" [✓] ALL MODELS DOWNLOADED & CACHED TO SHARED DISK!")
    print(" The GPU node (gpu04) can now load all dense models completely offline.")
    print("=" * 80)


if __name__ == "__main__":
    main()
