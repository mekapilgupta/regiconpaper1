import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PAPER1_DIR = Path(__file__).resolve().parent

# Load our 1000 query_ids from indic_msmarco_as.jsonl
with open(PAPER1_DIR / "data" / "indic_msmarco_as.jsonl", "r", encoding="utf-8") as f:
    target_records = [json.loads(line) for line in f if line.strip()]

target_ids = {r["query_id"]: r for r in target_records}
print(f"Target query IDs to recover: {len(target_ids)}")

recovered = {}

# Check MS MARCO v2.1 validation and train
for split in ["validation", "train", "test"]:
    if len(recovered) == len(target_ids):
        break
    print(f"Scanning MS MARCO v2.1 split: {split}...")
    try:
        ds = load_dataset("ms_marco", "v2.1", split=split)
        for row in tqdm(ds, desc=f"Indexing {split}"):
            qid = str(row["query_id"])
            if qid in target_ids and qid not in recovered:
                recovered[qid] = {
                    "query_id": qid,
                    "query_en": row["query"],
                    "query_type": row.get("query_type", ""),
                    "answers": row.get("answers", []),
                }
                if len(recovered) % 100 == 0:
                    print(f"[✓] Recovered {len(recovered)}/{len(target_ids)} English queries so far...")
    except Exception as e:
        print(f"Error on v2.1 {split}: {e}")

# If any still missing, check v1.1
if len(recovered) < len(target_ids):
    for split in ["validation", "train"]:
        if len(recovered) == len(target_ids):
            break
        print(f"Scanning MS MARCO v1.1 split: {split}...")
        try:
            ds = load_dataset("ms_marco", "v1.1", split=split)
            for row in tqdm(ds, desc=f"Indexing v1.1 {split}"):
                qid = str(row["query_id"])
                if qid in target_ids and qid not in recovered:
                    recovered[qid] = {
                        "query_id": qid,
                        "query_en": row["query"],
                        "query_type": row.get("query_type", ""),
                        "answers": row.get("answers", []),
                    }
        except Exception as e:
            print(f"Error on v1.1 {split}: {e}")

print(f"\n[✓] Final Recovery Count: {len(recovered)} / {len(target_ids)} English queries successfully recovered!")

# Save the master English query map
en_map_path = PAPER1_DIR / "data" / "msmarco_english_queries_map.json"
with open(en_map_path, "w", encoding="utf-8") as f:
    json.dump(recovered, f, indent=2, ensure_ascii=False)

print(f"[✓] Saved English queries map to {en_map_path}")
