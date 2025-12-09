# # build_adore_triplets.py
# import json
#
# with open("restricted_data/1200devTestSet.json") as f:
#     data = json.load(f)
#
# with open("adore_files/adore_train.jsonl", "w") as out:
#     for ex in data:
#         q = ex["question"]
#         # gold doc titles
#         gold = {d[0] for d in ex["supporting_facts"]}
#
#         positives, negatives = [], []
#         for title, sentences in ex["context"]:
#             doc_text = f"{title}: " + " ".join(sentences)
#             if title in gold:
#                 positives.append(doc_text)
#             else:
#                 negatives.append(doc_text)
#
#         out.write(json.dumps({
#             "query": q,
#             "positive_passages": positives,
#             "negative_passages": negatives
#         }) + "\n")


import json
import os

TRAIN_FILE = "adore_files/adore_train.jsonl"
PRE_DIR = "adore_preprocess"
os.makedirs(PRE_DIR, exist_ok=True)

queries = []
pids = []
rels = []
query_meta = []
pid_meta = []

with open(TRAIN_FILE, "r") as f:
    for line in f:
        item = json.loads(line.strip())
        q = item["query"]
        positives = item["positive_passages"]

        queries.append(q)
        query_meta.append(1)  # JSON int

        for p in positives:
            pids.append(p)
            rels.append(len(pids)-1)

        pid_meta.append(len(pids))  # cumulative count

# helper to write JSON arrays
def dump_json(name, data):
    with open(os.path.join(PRE_DIR, name), "w", encoding="utf-8") as f:
        json.dump(data, f)

dump_json("train-query", queries)
dump_json("train-query_meta", query_meta)
dump_json("train-pid", pids)
dump_json("train-pid_meta", pid_meta)
dump_json("train-rel", rels)

print("✔ Preprocess files written as valid JSON arrays.")
print("Your adore_preprocess directory is ready.")

