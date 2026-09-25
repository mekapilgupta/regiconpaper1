"""
Local Offline Neural Machine Translation Engine for Paper 1
Specialized for Southeast Asian languages (Indonesian & Thai) on MIRACL benchmark.
Supports:
1. NLLB-200 (facebook/nllb-200-distilled-600M)
2. OPUS-MT / MarianMT (Helsinki-NLP/opus-mt-id-en & en-id for Indonesian, opus-mt-th-en for Thai)
3. M2M100 (facebook/m2m100_418M) for direct cross-lingual translation
4. Sequential memory management with explicit GPU cache deallocation & live progress bars.
"""

import os
import sys
import gc
import re
from typing import List, Dict, Any, Optional
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

# NLLB-200 FLORES-200 language code mappings
NLLB_LANG_MAP = {
    "id": "ind_Latn",
    "th": "tha_Thai",
    "en": "eng_Latn",
}


def clean_mt_text(text: str) -> str:
    """Strips repeated dots, dashes, and whitespace artifacts from MT outputs."""
    if not text:
        return ""
    # Remove repeated dots or dashes like "........" or "- - - -"
    text = re.sub(r"[\.\-]{3,}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class LocalMTEngine:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _unload(self, model, tokenizer):
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def translate_nllb_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        model_id: str = "facebook/nllb-200-distilled-600M",
        batch_size: int = 32,
        desc: str = "NLLB-200",
    ) -> List[str]:
        """Translates a batch of texts using offline NLLB-200 distilled 600M."""
        print(f"[*] Loading NLLB ({model_id}) on {self.device} (local_files_only)...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=True,
        ).to(self.device)
        model.eval()

        src_flores = NLLB_LANG_MAP.get(src_lang, src_lang)
        tgt_flores = NLLB_LANG_MAP.get(tgt_lang, tgt_lang)
        tokenizer.src_lang = src_flores
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_flores)

        results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
                batch_texts = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)

                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=5,
                    do_sample=False,
                    max_length=64,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2,
                    early_stopping=True,
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend([clean_mt_text(d) for d in decoded])

        self._unload(model, tokenizer)
        return results

    def translate_opus_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 32,
        desc: str = "OPUS-MT",
    ) -> List[str]:
        """Translates texts using specialized Helsinki-NLP OPUS-MT models."""
        # Select appropriate model pair
        if src_lang == "id" and tgt_lang == "en":
            model_id = "Helsinki-NLP/opus-mt-id-en"
        elif src_lang == "en" and tgt_lang == "id":
            model_id = "Helsinki-NLP/opus-mt-en-id"
        elif src_lang == "th" and tgt_lang == "en":
            model_id = "Helsinki-NLP/opus-mt-th-en"
        elif src_lang == "en" and tgt_lang == "th":
            # For English -> Thai, use M2M100
            return self.translate_m2m100_batch(texts, src_lang="en", tgt_lang="th", batch_size=batch_size, desc="M2M100 (EN->TH)")
        else:
            raise ValueError(f"Unsupported OPUS-MT language pair: {src_lang} -> {tgt_lang}")

        print(f"[*] Loading OPUS-MT ({model_id}) on {self.device}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        model.eval()

        results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
                batch_texts = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)

                outputs = model.generate(
                    **inputs,
                    num_beams=4,
                    do_sample=False,
                    max_length=64,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend([clean_mt_text(d) for d in decoded])

        self._unload(model, tokenizer)
        return results

    def translate_m2m100_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        model_id: str = "facebook/m2m100_418M",
        batch_size: int = 32,
        desc: str = "M2M100",
    ) -> List[str]:
        """Translates texts using facebook/m2m100_418M."""
        print(f"[*] Loading M2M100 ({model_id}) on {self.device}...", flush=True)
        tokenizer = M2M100Tokenizer.from_pretrained(model_id)
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        model.eval()

        tokenizer.src_lang = src_lang
        forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)

        results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
                batch_texts = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)

                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=4,
                    do_sample=False,
                    max_length=64,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend([clean_mt_text(d) for d in decoded])

        self._unload(model, tokenizer)
        return results
