import json
import re
import string
from datetime import datetime
import faiss




###################################
# 2) LOAD ADORE RETRIEVER
###################################
# Requires:
#   adore_encoder = trained encoder from DRhard
#   adore_index = FAISS index of document embeddings
#   doc_ids.json = list of document strings in index order

# from adore import Encoder  # Provided by DRhard repo
#
# from evaluation_metrics import exact_match, cover_exact_match
# from generative_model_setup import answer_question
#
# print("Loading ADORE model and index...")
# adore_encoder = Encoder.load("adore_model")        # folder produced after training
# adore_index = faiss.read_index("adore_index.faiss") # built after encoding the corpus
# doc_ids = json.load(open("doc_ids.json", "r"))      # titles/text IDs in index order
#
#
# def adore_retrieve(question, k=5):
#     q_emb = adore_encoder.encode([question])
#     scores, idxs = adore_index.search(q_emb, k)
#     return [doc_ids[i] for i in idxs[0]]
#
#
# with open("1200devTestSet.json", "r") as f:
#     dataset = json.load(f)
#
# print(f"Loaded {len(dataset)} evaluation queries.")
#
#
#
# results = []
#
# for ex in dataset:
#     q = ex["question"]
#     gold = ex["answer"]
#
#     adore_docs = adore_retrieve(q, k=5)
#     context = "\n\n".join(adore_docs)
#
#     pred = answer_question(q, context)
#
#     # scoring
#     em = exact_match(pred, gold)
#     cem = cover_exact_match(pred, gold)
#
#     result_entry = {
#         "id": ex["id"],
#         "question": q,
#         "gold_answer": gold,
#         "predicted_answer": pred,
#         "contexts_used": adore_docs,
#         "EM": em,
#         "CEM": cem,
#         "retriever": "ADORE",
#         "model": "llama3.1:8b",
#         "timestamp": datetime.now().isoformat()
#     }
#
#     results.append(result_entry)
#     print(f"[{ex['id']}] EM={em}, CEM={cem}")
#
# with open("experiment_results/adore_results.json", "w") as f:
#     json.dump(results, f, indent=2)
#
# print("\nADORE QA Experiment Complete.")
# print("Saved results to adore_results.json")




import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

########################################
# CONFIGURATION
########################################

DATA_PATH = "1200devTestSet.json"         # Your dataset
PREPROCESS_DIR = "adore_preprocess"        # ADORE expects a preprocess directory
PEMBED_PATH = "passages.memmap"            # Document embeddings file for ADORE
INIT_PATH = "contriever_init"              # Folder to store contriever model
MODEL_SAVE_DIR = "adore_model"             # Output directory for trained ADORE model

########################################
# STEP 1 — Download Contriever model
########################################

print("Loading Contriever model...")
model = SentenceTransformer("facebook/contriever-msmarco")

if not os.path.exists(INIT_PATH):
    os.makedirs(INIT_PATH)
    model.save(INIT_PATH)  # Saves the contriever weights so ADORE can use them
    print(f"Saved Contriever init model to {INIT_PATH}")

########################################
# STEP 2 — Build passage embeddings
########################################

print("Building passage embeddings (pembed)...")

docs = []
ids = []

with open(DATA_PATH) as f:
    data = json.load(f)
    for ex in data:
        for title, sents in ex["context"]:
            text = " ".join(sents)
            docs.append(text)
            ids.append(title)

# Remove duplicate docs while preserving order
unique_docs = list(dict.fromkeys(docs))
unique_ids = list(dict.fromkeys(ids))

print(f"Total unique documents: {len(unique_docs)}")

# Encode all docs to embeddings
embs = model.encode(unique_docs, convert_to_numpy=True, show_progress_bar=True)

# Save memmap
fp = np.memmap(PEMBED_PATH, dtype="float32", mode="w+", shape=embs.shape)
fp[:] = embs[:]
fp.flush()

json.dump(unique_ids, open("doc_ids.json", "w"))
print(f"Saved document embeddings to {PEMBED_PATH}")
print("Saved doc_ids.json")


########################################
# STEP 3 — PREPROCESS DIRECTORY SETUP
########################################

if not os.path.exists(PREPROCESS_DIR):
    os.makedirs(PREPROCESS_DIR)

# ADORE only checks for existence of this folder
with open(os.path.join(PREPROCESS_DIR, "dummy.txt"), "w") as f:
    f.write("placeholder")

print(f"Created preprocess dir at {PREPROCESS_DIR}")


########################################
# STEP 4 — TRAIN ADORE
########################################

print("Starting ADORE training...")
cmd = f"""
py -3.9 DRhard\\adore\\train.py ^
  --metric_cut 200 ^
  --init_path {INIT_PATH} ^
  --pembed_path {PEMBED_PATH} ^
  --model_save_dir {MODEL_SAVE_DIR} ^
  --log_dir adore_logs ^
  --preprocess_dir {PREPROCESS_DIR} ^
  --num_train_epochs 1 ^
  --per_gpu_batch_size 4 ^
  --learning_rate 2e-5
"""

print("RUN THE FOLLOWING COMMAND IN POWERSHELL:")
print(cmd)

print("\nOnce finished, ADORE model will be saved in:", MODEL_SAVE_DIR)
