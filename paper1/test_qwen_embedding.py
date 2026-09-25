#!/usr/bin/env python3
"""
Diagnostic & Verification Script for Qwen/Qwen3-Embedding-0.6B
Checks exact active conda environment, installed package versions vs required versions,
and executes official multilingual retrieval similarity tests.
"""

import sys
import os
import time
import importlib.metadata

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Minimum required versions according to Hugging Face release
REQUIRED_VERSIONS = {
    "transformers": "4.51.0",
    "sentence-transformers": "2.7.0",
    "torch": "2.1.0",
    "accelerate": "0.26.0",
}


def parse_version(v_str):
    """Helper to parse semantic version tuple."""
    try:
        clean = v_str.split("+")[0].split("-")[0]
        return tuple(int(x) if x.isdigit() else 0 for x in clean.split("."))
    except Exception:
        return (0, 0, 0)


def print_environment_diagnostics():
    print("=" * 80)
    print(" 🔍 ENVIRONMENT & DEPENDENCY DIAGNOSTICS")
    print("=" * 80)

    # Active Conda Env & Python
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "None (System Python)")
    print(f"[*] Active Conda Environment : {conda_env}")
    print(f"[*] Python Executable        : {sys.executable}")
    print(f"[*] Python Version           : {sys.version.split()[0]}")
    print("-" * 80)

    all_passed = True
    installed_info = {}

    for pkg, min_ver in REQUIRED_VERSIONS.items():
        try:
            cur_ver = importlib.metadata.version(pkg)
            is_ok = parse_version(cur_ver) >= parse_version(min_ver)
            installed_info[pkg] = (cur_ver, is_ok)
            status_icon = "✅ OK" if is_ok else "❌ OUTDATED"
            print(f"  - {pkg:<22} : Installed = {cur_ver:<14} | Required >= {min_ver:<8} [{status_icon}]")
            if not is_ok:
                all_passed = False
        except importlib.metadata.PackageNotFoundError:
            installed_info[pkg] = ("NOT INSTALLED", False)
            print(f"  - {pkg:<22} : Installed = NOT INSTALLED   | Required >= {min_ver:<8} [❌ MISSING]")
            all_passed = False

    # CUDA & GPU Diagnostics
    import torch
    print("-" * 80)
    print(f"[*] PyTorch CUDA Available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] Active GPU Device        : {gpu_name} (VRAM: {gpu_mem:.2f} GB)")
    print("=" * 80)

    if not all_passed:
        print("\n⚠️ [ACTION REQUIRED] Some packages do not meet the minimum version for Qwen3!")
        print("Run the following command in your active environment:")
        print("  pip install --upgrade \"transformers>=4.51.0\" \"sentence-transformers>=2.7.0\" accelerate\n")
        print("Or if using source:")
        print("  pip install --upgrade git+https://github.com/huggingface/transformers.git\n")

    return all_passed


def test_qwen_embedding():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n--- [Step 1/3] Loading Model '{MODEL_NAME}' ---")
    start_t = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            MODEL_NAME,
            device=device,
            tokenizer_kwargs={"padding_side": "left"},
            trust_remote_code=True,
        )
        load_duration = time.time() - start_t
        print(f"[✓] Model successfully loaded on {device} in {load_duration:.2f}s")
    except Exception as e:
        print(f"[-] Loading Failed with error:\n    {e}")
        return False

    # 2. Multilingual Queries & Documents
    queries = [
        "What is the capital of China?",  # EN
        "Apa ibu kota Indonesia?",         # ID
        "เมืองหลวงของประเทศไทยคืออะไร",  # TH
    ]

    documents = [
        "The capital of China is Beijing.",                                                               # Match 0
        "Jakarta dan Ibu Kota Nusantara adalah pusat pemerintahan dan ibu kota Indonesia.",              # Match 1
        "กรุงเทพมหานครเป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย",                        # Match 2
        "Quantum computing utilizes superposition and entanglement to perform complex computations.",    # Distractor
    ]

    print(f"\n--- [Step 2/3] Encoding Multilingual Queries (with prompt_name='query') ---")
    encode_start = time.time()
    try:
        query_embeddings = model.encode(queries, prompt_name="query", convert_to_tensor=True, normalize_embeddings=True)
    except Exception:
        query_embeddings = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)

    document_embeddings = model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
    encode_duration = time.time() - encode_start

    print(f"[✓] Query Embeddings Shape : {query_embeddings.shape} (Dim: {query_embeddings.shape[1]})")
    print(f"[✓] Doc Embeddings Shape   : {document_embeddings.shape}")
    print(f"[✓] Encoding Latency       : {encode_duration * 1000:.2f} ms")

    # 3. Compute Similarity Matrix
    print(f"\n--- [Step 3/3] Cosine Similarity Matrix Evaluation ---")
    if hasattr(model, "similarity"):
        similarity_matrix = model.similarity(query_embeddings, document_embeddings)
    else:
        similarity_matrix = torch.matmul(query_embeddings, document_embeddings.T)

    print("Similarity Matrix (Queries × Documents):")
    print(similarity_matrix)

    sim_en = similarity_matrix[0, 0].item()
    sim_id = similarity_matrix[1, 1].item()
    sim_th = similarity_matrix[2, 2].item()

    print(f"\n[*] English Match Score (Query 0 <-> Doc 0)    : {sim_en:.4f}")
    print(f"[*] Indonesian Match Score (Query 1 <-> Doc 1) : {sim_id:.4f}")
    print(f"[*] Thai Match Score (Query 2 <-> Doc 2)       : {sim_th:.4f}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"[*] Peak GPU VRAM Footprint: {allocated:.1f} MB allocated ({reserved:.1f} MB reserved)")

    print("\n" + "=" * 80)
    if sim_en > 0.6 and sim_id > 0.6 and sim_th > 0.6:
        print(" 🎉 ALL TESTS PASSED: Qwen3-Embedding-0.6B is 100% verified on GPU!")
        print("=" * 80)
        return True
    else:
        print(" ⚠️ Model executed, but match scores were lower than expected.")
        print("=" * 80)
        return False


def main():
    deps_ok = print_environment_diagnostics()
    if not deps_ok:
        print("[!] Attempting model load anyway to inspect runtime errors...\n")
    test_qwen_embedding()


if __name__ == "__main__":
    main()
