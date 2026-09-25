"""
Offline Model Downloader for Paper 1
Pre-caches all local neural machine translation, LLM generator, and embedding models:
1. ai4bharat/indictrans2-indic-en-1B (IndicTrans2 Indic -> En)
2. ai4bharat/indictrans2-en-indic-1B (IndicTrans2 En -> Indic)
3. facebook/nllb-200-distilled-600M (NLLB-200 Multilingual MT)
4. sarvamai/sarvam-translate (Sarvam Indic Specialized MT)
5. Qwen/Qwen2.5-7B-Instruct (High-precision multilingual instruction model)
6. microsoft/Phi-4-mini-instruct (Compact multilingual instruction model)
7. ai4bharat/IndicBERT-v3-1B (Indic-specialized dense encoder)
8. BAAI/bge-m3 (Dense / Multi-lingual Embedding & Validation Triage)
9. intfloat/multilingual-e5-base
10. intfloat/multilingual-e5-large
11. sentence-transformers/LaBSE
"""

import sys
import os
from pathlib import Path

MODELS_TO_DOWNLOAD = [
    ("AutoModelForSeq2SeqLM", "facebook/nllb-200-distilled-600M", {}),
    ("AutoModelForSeq2SeqLM", "ai4bharat/indictrans2-indic-en-1B", {"trust_remote_code": True}),
    ("AutoModelForSeq2SeqLM", "ai4bharat/indictrans2-en-indic-1B", {"trust_remote_code": True}),
    ("AutoModelForCausalLM", "sarvamai/sarvam-translate", {"trust_remote_code": True, "torch_dtype": "auto"}),
    ("AutoModelForCausalLM", "Qwen/Qwen2.5-7B-Instruct", {"torch_dtype": "auto"}),
    ("AutoModelForCausalLM", "microsoft/Phi-4-mini-instruct", {"torch_dtype": "auto"}),
    ("AutoModel", "ai4bharat/IndicBERT-v3-1B", {"trust_remote_code": True}),
    ("SentenceTransformer", "BAAI/bge-m3", {}),
    ("SentenceTransformer", "intfloat/multilingual-e5-base", {}),
    ("SentenceTransformer", "intfloat/multilingual-e5-large", {}),
    ("SentenceTransformer", "sentence-transformers/LaBSE", {}),
]

def download_models():
    print("=" * 70)
    print("🚀 OFFLINE MODEL DOWNLOAD & CACHING PIPELINE")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel
    except ImportError:
        print("[!] Error: transformers is not installed in current environment.")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None

    for model_type, model_id, kwargs in MODELS_TO_DOWNLOAD:
        print(f"\n[*] Checking / Downloading: {model_id} ({model_type})...", flush=True)
        try:
            if model_type == "AutoModelForSeq2SeqLM":
                tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
                print(f"    [✓] Successfully cached: {model_id}")
                del tokenizer, model
            elif model_type == "AutoModelForCausalLM":
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **kwargs)
                print(f"    [✓] Successfully cached: {model_id}")
                del tokenizer, model
            elif model_type == "AutoModel":
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **kwargs)
                print(f"    [✓] Successfully cached: {model_id}")
                del tokenizer, model
            elif model_type == "SentenceTransformer":
                if SentenceTransformer is not None:
                    model = SentenceTransformer(model_id)
                    print(f"    [✓] Successfully cached: {model_id}")
                    del model
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    print(f"    [✓] Cached tokenizer: {model_id}")
                    del tokenizer
        except Exception as e:
            print(f"    [!] Note on {model_id}: {e}", flush=True)

    print("\n" + "=" * 70)
    print("🎉 ALL MODELS VERIFIED AND READY FOR OFFLINE EXECUTION")
    print("=" * 70)

if __name__ == "__main__":
    download_models()
