"""
Ingest MIRACL dataset for Southeast Asian languages: Indonesian (id) and Thai (th).
Extracts:
1. Canonical benchmark queries (300 per language) with verified positive passages.
2. Full passage corpus for BM25 and dense retrieval indexing.
"""

import os
import gzip
import json
import time
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, SPLITS_DIR

CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

MIRACL_CORPUS_FILES = {
    "th": [
        "miracl-corpus-v1.0-th/docs-0.jsonl.gz",
        "miracl-corpus-v1.0-th/docs-1.jsonl.gz",
    ],
    "id": [
        "miracl-corpus-v1.0-id/docs-0.jsonl.gz",
        "miracl-corpus-v1.0-id/docs-1.jsonl.gz",
        "miracl-corpus-v1.0-id/docs-2.jsonl.gz",
    ],
}


def download_and_prepare_miracl(languages=["id", "th"], sample_size=300):
    print("=" * 75)
    print(f"📥 INGESTING MIRACL BENCHMARK FOR SOUTHEAST ASIAN LANGUAGES: {languages}")
    print("=" * 75)

    for lang in languages:
        print(f"\n--- [Processing Language: {lang.upper()}] ---", flush=True)

        # 1. Download topics and qrels
        print(f"[*] Downloading MIRACL {lang} topics & qrels...", flush=True)
        topics_file = hf_hub_download(
            repo_id="miracl/miracl",
            filename=f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv",
            repo_type="dataset",
        )
        qrels_file = hf_hub_download(
            repo_id="miracl/miracl",
            filename=f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv",
            repo_type="dataset",
        )

        df_topics = pd.read_csv(topics_file, sep="\t", names=["query_id", "query"])
        df_qrels = pd.read_csv(qrels_file, sep="\t", names=["query_id", "iteration", "doc_id", "relevance"])

        # Filter positive qrels
        df_pos_qrels = df_qrels[df_qrels["relevance"] > 0]
        pos_map = {}
        for _, row in df_pos_qrels.iterrows():
            qid = str(row["query_id"])
            docid = str(row["doc_id"])
            if qid not in pos_map:
                pos_map[qid] = []
            pos_map[qid].append(docid)

        target_qids = [str(qid) for qid in df_topics["query_id"] if str(qid) in pos_map][:sample_size]
        print(f"[✓] Identified {len(target_qids)} benchmark queries with positive relevance judgments.")

        # Needed docids for positive passages
        needed_pos_docids = set()
        for qid in target_qids:
            needed_pos_docids.update(pos_map[qid])

        # 2. Download and read corpus files
        corpus_gz_files = []
        for cfile in MIRACL_CORPUS_FILES[lang]:
            print(f"[*] Downloading {cfile}...", flush=True)
            local_gz = hf_hub_download(
                repo_id="miracl/miracl-corpus",
                filename=cfile,
                repo_type="dataset",
            )
            corpus_gz_files.append(local_gz)

        # Extract needed positive passages and build clean corpus file
        print(f"[*] Scanning corpus to extract positive passages & export corpus...", flush=True)
        pos_passages_text = {}
        corpus_out_path = CORPUS_DIR / f"corpus_{lang}.jsonl"

        total_corpus_docs = 0
        with open(corpus_out_path, "w", encoding="utf-8") as out_f:
            for gz_path in corpus_gz_files:
                with gzip.open(gz_path, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        if not line.strip():
                            continue
                        doc = json.loads(line)
                        docid = str(doc["docid"])
                        title = doc.get("title", "").strip()
                        text = doc.get("text", "").strip()
                        full_passage = f"{title}. {text}".strip() if title else text

                        clean_doc = {
                            "doc_id": docid,
                            "title": title,
                            "passage": full_passage,
                        }
                        out_f.write(json.dumps(clean_doc, ensure_ascii=False) + "\n")
                        total_corpus_docs += 1

                        if docid in needed_pos_docids and docid not in pos_passages_text:
                            pos_passages_text[docid] = full_passage

        print(f"[✓] Saved full {lang.upper()} corpus: {corpus_out_path} ({total_corpus_docs:,} passages).")

        # 3. Assemble Canonical Pilot Queries
        canonical_records = []
        queries_dict = dict(zip(df_topics["query_id"].astype(str), df_topics["query"]))

        for qid in target_qids:
            query_text = queries_dict[qid]
            pos_docids = pos_map[qid]
            # Find the best positive passage text
            pos_text = ""
            best_docid = pos_docids[0]
            for did in pos_docids:
                if did in pos_passages_text and len(pos_passages_text[did]) > 20:
                    pos_text = pos_passages_text[did]
                    best_docid = did
                    break

            if not pos_text:
                continue

            record = {
                "query_id": qid,
                "language": lang,
                "q_ref": query_text,
                "positive_passage_id": best_docid,
                "positive_passage": pos_text,
                "all_positive_docids": pos_docids,
                "dataset_source": "miracl/miracl",
            }
            canonical_records.append(record)
            if len(canonical_records) >= sample_size:
                break

        split_path = SPLITS_DIR / f"canonical_pilot_{lang}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for r in canonical_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[✓] Exported {len(canonical_records)} canonical queries to {split_path}")

    print("\n" + "=" * 75)
    print("✅ MIRACL INGESTION COMPLETE FOR THAI & INDONESIAN!")
    print("=" * 75)


if __name__ == "__main__":
    download_and_prepare_miracl(languages=["id", "th"], sample_size=300)
