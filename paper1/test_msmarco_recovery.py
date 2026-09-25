from datasets import load_dataset

# Let's search MS MARCO QnA v2.1 dev and train
target_ids = {"10219", "10102", "9936", "11383", "10216", "12250", "10875"}

for config in ["v2.1", "v1.1"]:
    for split in ["dev", "train", "test"]:
        try:
            print(f"Checking {config} split {split}...")
            ds = load_dataset("ms_marco", config, split=split, streaming=True)
            for i, row in enumerate(ds):
                qid = str(row["query_id"])
                if qid in target_ids:
                    print(f"FOUND in {config} {split}: ID {qid} -> {row['query']}")
                if i > 15000:
                    break
        except Exception as e:
            print(f"Error on {config} {split}: {e}")
