"""
Dataset Loader, Split Manager, and Corpus Deduplicator
Paper 1: Evaluation Invariance under Multilingual Benchmark Transformation
Handles Assamese (as) and Bengali (bn), genuine MS MARCO English recovery, and MD5 text-hash corpus deduplication.
"""

import os
import re
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

from config import (
    DATA_DIR,
    SPLITS_DIR,
    EXPERIMENT_SEED,
    DEFAULT_LANGUAGES,
    PILOT_SAMPLE_SIZE,
    FINAL_SAMPLE_SIZE,
)


class IndicMSMARCOLoader:
    def __init__(self, data_dir: Optional[Path] = None, seed: int = EXPERIMENT_SEED):
        self.data_dir = data_dir or DATA_DIR
        self.splits_dir = SPLITS_DIR
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(self.seed)

        self.en_map_path = self.data_dir / "msmarco_english_queries_map.json"
        self.en_map: Dict[str, Dict[str, Any]] = {}
        if self.en_map_path.exists():
            with open(self.en_map_path, "r", encoding="utf-8") as f:
                self.en_map = json.load(f)
            print(f"[*] Loaded {len(self.en_map)} genuine MS MARCO English query mappings.")

    def _get_passage_hash_id(self, passage_text: str) -> str:
        """Returns deterministic 12-char MD5 doc ID for normalized passage text."""
        normalized = re.sub(r"\s+", " ", passage_text.strip().lower())
        h = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
        return f"doc_{h}"

    def load_and_enrich_dataset(self, language: str) -> List[Dict[str, Any]]:
        """
        Loads cached IndicMSMARCO, injects genuine MS MARCO English queries,
        and assigns deduplicated MD5 passage document IDs.
        """
        cache_file = self.data_dir / f"indic_msmarco_{language}.jsonl"
        if not cache_file.exists():
            raise FileNotFoundError(f"Missing dataset file: {cache_file}")

        records = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        enriched = []
        for r in records:
            qid = str(r.get("query_id", r.get("id", "")))
            q_ref = str(r.get("query", r.get("q_ref", ""))).strip()
            passage = str(r.get("positive_passage", r.get("passage", ""))).strip()
            answer = str(r.get("answer", "")).strip()

            # 1. Recover genuine English query from canonical MS MARCO map
            en_info = self.en_map.get(qid, {})
            q_en = en_info.get("query_en", "").strip()
            q_type = en_info.get("query_type", r.get("query_type", "DESCRIPTION"))
            answers = en_info.get("answers", [answer] if answer else [])

            # 2. Compute canonical deduplicated passage doc_id
            doc_id = self._get_passage_hash_id(passage)

            item = {
                "query_id": qid,
                "language": language,
                "q_en_source": q_en,  # Genuine MS MARCO English query (never back-translated)
                "q_ref": q_ref,
                "positive_passage_id": doc_id,
                "positive_passage": passage,
                "answer": answers[0] if answers else answer,
                "query_type": q_type,
                "dataset_source": "ai4bharat/IndicMSMARCO",
                "is_genuine_english": bool(q_en),
            }
            enriched.append(item)

        print(f"[*] Processed {len(enriched)} '{language}' records. (Genuine English available: {sum(1 for x in enriched if x['is_genuine_english'])}/{len(enriched)})")
        return enriched

    def build_pilot_splits(
        self,
        sample_size: int = PILOT_SAMPLE_SIZE,
        languages: List[str] = DEFAULT_LANGUAGES,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], Dict[str, Dict[str, int]]]:
        """
        Creates deterministic pilot split of N queries per language,
        builds a deduplicated shared corpus of passages, and builds qrels mapping.
        """
        data_by_lang = {}
        for lang in languages:
            data_by_lang[lang] = self.load_and_enrich_dataset(lang)

        # Find query IDs present in both languages that have genuine English queries
        valid_id_sets = [
            set(r["query_id"] for r in records if r["is_genuine_english"])
            for records in data_by_lang.values()
        ]
        common_valid_ids = sorted(list(set.intersection(*valid_id_sets))) if valid_id_sets else []

        rng = random.Random(self.seed)
        shuffled_ids = list(common_valid_ids)
        rng.shuffle(shuffled_ids)

        pilot_selected_ids = set(shuffled_ids[:min(sample_size, len(shuffled_ids))])
        print(f"[*] Selected {len(pilot_selected_ids)} common pilot query IDs for languages {languages}")

        pilot_data: Dict[str, List[Dict[str, Any]]] = {}
        corpus: Dict[str, str] = {}  # doc_id -> passage text
        qrels: Dict[str, Dict[str, int]] = {}  # query_id -> {doc_id: 1}

        for lang, records in data_by_lang.items():
            pilot_records = [r for r in records if r["query_id"] in pilot_selected_ids]
            pilot_data[lang] = pilot_records

            # Populate unified corpus and qrels
            for r in pilot_records:
                qid = r["query_id"]
                doc_id = r["positive_passage_id"]
                passage_text = r["positive_passage"]

                corpus[doc_id] = passage_text
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = 1

            # Save split to jsonl
            split_out = self.splits_dir / f"canonical_pilot_{lang}.jsonl"
            with open(split_out, "w", encoding="utf-8") as f:
                for r in pilot_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Save unified corpus and qrels
        with open(self.splits_dir / "pilot_corpus.json", "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        with open(self.splits_dir / "pilot_qrels.json", "w", encoding="utf-8") as f:
            json.dump(qrels, f, indent=2)

        print(f"[✓] Pilot dataset prepared:")
        print(f"    - Queries per language : {len(pilot_selected_ids)}")
        print(f"    - Unique corpus docs   : {len(corpus)}")
        print(f"    - Total qrels          : {len(qrels)}")

        return pilot_data, corpus, qrels
