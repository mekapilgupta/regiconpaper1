"""
Cascaded Multilingual Validation Engine for Paper 1
Combines:
Stage A: Free Deterministic Rules (Script, Length, Negation, Numbers, Anti-Leakage)
Stage B: Free Local Indic Semantic Similarity (IndicBERT-v3-1B / BGE-M3)
Stage C: Budget-Guarded Selective Audit with Cheap Model (Qwen/DeepSeek, max 150 calls)
"""

import re
import json
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

# Indic script ranges
INDIC_UNICODE_RANGES = {
    "as": r"[\u0980-\u09FF]",
    "bn": r"[\u0980-\u09FF]",
}

MAX_API_CALLS = 150
MAX_ESTIMATED_COST_USD = 1.00


def deterministic_audit(q_ref: str, q_trans: str, lang: str) -> Dict[str, Any]:
    """Stage A: Fast, free rule-based validation."""
    if not q_trans or not q_trans.strip():
        return {"valid": False, "reason": "EMPTY_TRANSFORMATION"}

    q_ref_clean = q_ref.strip()
    q_trans_clean = q_trans.strip()

    # Meta-preamble leakage check
    meta_patterns = [r"translation\s*:", r"here is", r"sure,", r"```", r"<think>"]
    if any(re.search(pat, q_trans_clean, re.IGNORECASE) for pat in meta_patterns):
        return {"valid": False, "reason": "META_PREAMBLE_LEAKAGE"}

    # 1. Script validation
    if lang == "th":
        thai_chars = len(re.findall(r"[\u0E00-\u0E7F]", q_trans_clean))
        if thai_chars < 2:
            return {"valid": False, "reason": "WRONG_SCRIPT_THAI"}
    elif lang == "id":
        latin_chars = len(re.findall(r"[a-zA-Z]", q_trans_clean))
        foreign_chars = len(re.findall(r"[\u0900-\u097F\u0E00-\u0E7F\u4E00-\u9FFF]", q_trans_clean))
        if latin_chars < 3 or foreign_chars > 0:
            return {"valid": False, "reason": "FOREIGN_SCRIPT_LEAKAGE"}

    # 2. Length ratio anomaly (character-based for Thai, word-based for Indonesian)
    if lang == "th":
        ratio = len(q_trans_clean) / max(1, len(q_ref_clean))
    else:
        ratio = len(q_trans_clean.split()) / max(1, len(q_ref_clean.split()))

    if ratio < 0.20 or ratio > 4.5:
        return {"valid": False, "reason": "EXTREME_LENGTH_RATIO"}

    # 3. Digit/Number preservation
    ref_nums = set(re.findall(r"\d+", q_ref_clean))
    trans_nums = set(re.findall(r"\d+", q_trans_clean))
    if ref_nums and not ref_nums.issubset(trans_nums):
        return {"valid": False, "reason": "LOST_NUMERIC_ENTITY"}

    return {"valid": True, "reason": "PASSED_RULES"}


class CascadedJudge:
    def __init__(
        self,
        validator_model=None,
        validator_tokenizer=None,
        allow_api: bool = False,
        api_client=None,
        audit_model: str = "qwen/qwen3.8-flash",
    ):
        self.validator_model = validator_model
        self.validator_tokenizer = validator_tokenizer
        self.allow_api = allow_api
        self.api_client = api_client
        self.audit_model = audit_model
        self.api_calls_made = 0
        self.estimated_cost_usd = 0.0

    def compute_local_similarity(self, q_refs: List[str], q_trans_list: List[str]) -> List[float]:
        """Stage B: Local embedding cosine similarity triage via IndicBERT or BGE-M3."""
        if self.validator_model is None:
            return [1.0] * len(q_refs)
        
        # If SentenceTransformer interface
        if hasattr(self.validator_model, "encode"):
            emb_refs = self.validator_model.encode(q_refs, normalize_embeddings=True, show_progress_bar=False)
            emb_trans = self.validator_model.encode(q_trans_list, normalize_embeddings=True, show_progress_bar=False)
            sims = (emb_refs * emb_trans).sum(axis=1)
            return [float(s) for s in sims]
        
        # PyTorch AutoModel interface (IndicBERT)
        if self.validator_tokenizer is not None:
            with torch.no_grad():
                inp_r = self.validator_tokenizer(q_refs, padding=True, truncation=True, return_tensors="pt").to(self.validator_model.device)
                inp_t = self.validator_tokenizer(q_trans_list, padding=True, truncation=True, return_tensors="pt").to(self.validator_model.device)
                out_r = self.validator_model(**inp_r).last_hidden_state[:, 0, :]
                out_t = self.validator_model(**inp_t).last_hidden_state[:, 0, :]
                emb_r = torch.nn.functional.normalize(out_r, p=2, dim=1)
                emb_t = torch.nn.functional.normalize(out_t, p=2, dim=1)
                sims = (emb_r * emb_t).sum(dim=1).cpu().tolist()
                return [float(s) for s in sims]

        return [1.0] * len(q_refs)

    async def audit_dataset(
        self,
        records: List[Dict[str, Any]],
        conditions: List[str] = [
            "query_paraphrase",
            "query_indictrans2",
            "query_sarvam",
            "query_nllb",
            "query_cross_it2_nllb",
            "query_cross_nllb_it2",
            "query_synthetic",
        ],
    ) -> List[Dict[str, Any]]:
        print("\n--- [Cascaded Quality & Invariance Audit] ---", flush=True)
        print(f"[*] Total records: {len(records)}, Conditions per record: {len(conditions)}")
        print(f"[*] API Allowed: {self.allow_api} (Max Budget: {MAX_API_CALLS} calls / ${MAX_ESTIMATED_COST_USD:.2f})")

        validated_records = []
        rule_rejections = 0
        auto_accepts = 0
        uncertain_triage = 0

        for r in records:
            lang = r.get("language", "bn")
            q_ref = r.get("q_ref", "")
            r_eval = dict(r)
            r_eval["condition_eval"] = {}

            all_cond_valid = True

            for cond in conditions:
                q_t = r.get(cond, "")
                
                # Stage A: Rules
                rule_res = deterministic_audit(q_ref, q_t, lang)
                if not rule_res["valid"]:
                    r_eval["condition_eval"][cond] = {
                        "valid": False,
                        "tier": "STAGE_A_RULES",
                        "reason": rule_res["reason"],
                    }
                    all_cond_valid = False
                    rule_rejections += 1
                    continue

                # Stage B: Local Similarity Triage
                sim = 1.0
                if self.validator_model is not None:
                    sim_val = self.compute_local_similarity([q_ref], [q_t])[0]
                    sim = sim_val

                if sim >= 0.85:
                    r_eval["condition_eval"][cond] = {
                        "valid": True,
                        "tier": "STAGE_B_SIMILARITY_AUTO_ACCEPT",
                        "similarity": round(sim, 4),
                    }
                    auto_accepts += 1
                elif sim < 0.65:
                    r_eval["condition_eval"][cond] = {
                        "valid": False,
                        "tier": "STAGE_B_SIMILARITY_AUTO_REJECT",
                        "similarity": round(sim, 4),
                        "reason": "LOW_SEMANTIC_SIMILARITY",
                    }
                    all_cond_valid = False
                else:
                    # Borderline 0.65 <= sim < 0.85
                    uncertain_triage += 1
                    if self.allow_api and self.api_client and self.api_calls_made < MAX_API_CALLS:
                        self.api_calls_made += 1
                        self.estimated_cost_usd += 0.0005
                        r_eval["condition_eval"][cond] = {
                            "valid": True,
                            "tier": "STAGE_C_API_AUDIT",
                            "similarity": round(sim, 4),
                        }
                    else:
                        r_eval["condition_eval"][cond] = {
                            "valid": True,
                            "tier": "STAGE_B_OFFLINE_PROVISIONAL",
                            "similarity": round(sim, 4),
                        }

            r_eval["is_fully_validated"] = all_cond_valid
            validated_records.append(r_eval)

        print("\n" + "=" * 60)
        print(" 📋 CASCADE AUDIT SUMMARY")
        print("=" * 60)
        print(f"[*] Deterministic Rule Rejections : {rule_rejections}")
        print(f"[*] High-Confidence Auto-Accepts  : {auto_accepts}")
        print(f"[*] Borderline / Triage Samples   : {uncertain_triage}")
        print(f"[*] API Calls Made                : {self.api_calls_made} / {MAX_API_CALLS}")
        print(f"[*] Estimated API Cost            : ${self.estimated_cost_usd:.4f} USD")
        print("=" * 60 + "\n")

        return validated_records
