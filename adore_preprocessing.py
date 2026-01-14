import json
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from dexter.data.datastructures.evidence import Evidence
from dexter.retriever.dense.Contriever import Contriever
from dexter.retriever.dense.DenseFullSearch import DenseHyperParams
from transformers import AutoTokenizer

from DRhard.adore.model import RobertaDot

CORPUS_PATH = "raw_data/wiki_musique_corpus.json"  # id -> {title, text}
OUT_BASE = "data/passage"
MODEL_NAME = "facebook/contriever"

BATCH_SIZE = 16


def ensure_dirs():
    os.makedirs(f"{OUT_BASE}/preprocess", exist_ok=True)
    os.makedirs(f"{OUT_BASE}/evaluate/contriever", exist_ok=True)


def convert_to_evidence(path: str) -> List[Evidence]:
    evidences = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for doc_id, doc in data.items():
        title = doc.get("title", "")
        text = doc["text"]
        evidences.append(Evidence(text, doc_id, title))

    print(f"Loaded {len(evidences)} documents")
    return evidences


def save_passages_json(evidences: List[Evidence]):
    passages = {}
    for ev in evidences:
        passages[ev.id()] = {
            "text": f"{ev.title()} [SEP] {ev.text()}" if ev.title() else ev.text()
        }

    out_path = f"{OUT_BASE}/preprocess/passages.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2)

    print(f"Saved passages.json → {out_path}")


def encode_with_dexter(evidences: List[Evidence]) -> torch.Tensor:
    cfg = DenseHyperParams(
        query_encoder_path=MODEL_NAME,
        document_encoder_path=MODEL_NAME,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )

    retriever = Contriever(cfg)
    retriever.context_encoder.eval()
    with torch.no_grad():
        embeddings = retriever.encode_corpus(evidences)

    print("Embeddings shape:", embeddings.shape)
    return embeddings


def save_memmap(embeddings: torch.Tensor, evidences: List[Evidence]):
    emb = embeddings.cpu().numpy().astype("float32")

    memmap_path = f"{OUT_BASE}/evaluate/contriever/passages.memmap"
    meta_path = f"{OUT_BASE}/evaluate/contriever/passages_meta.json"

    mmap = np.memmap(
        memmap_path,
        dtype="float32",
        mode="w+",
        shape=emb.shape
    )
    mmap[:] = emb
    mmap.flush()

    meta = {
        "num_passages": emb.shape[0],
        "dim": emb.shape[1],
        "doc_ids": [ev.id() for ev in evidences]
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved passages.memmap → {memmap_path}")
    print(f"Saved passages_meta.json → {meta_path}")


def create_placeholders():
    queries_path = f"{OUT_BASE}/preprocess/queries.train.json"
    qrels_path = f"{OUT_BASE}/preprocess/qrels.train.tsv"

    if not os.path.exists(queries_path):
        with open(queries_path, "w") as f:
            json.dump({}, f)

    if not os.path.exists(qrels_path):
        open(qrels_path, "w").close()

    print("Created empty queries.train.json and qrels.train.tsv")


def main():
    ensure_dirs()

    evidences = convert_to_evidence(CORPUS_PATH)

    save_passages_json(evidences)

    embeddings = encode_with_dexter(evidences)

    norms = embeddings.norm(dim=1)
    print("Embedding norms:", norms.min().item(), norms.max().item())

    save_memmap(embeddings, evidences)

    create_placeholders()

    print("\nAll ADORE-required files generated successfully.")


def setup_training_files():
    TRAIN_PATH = "raw_data/train.json"
    OUT_DIR = "data/passage/preprocess"

    os.makedirs(OUT_DIR, exist_ok=True)

    CORPUS_PATH = "raw_data/wiki_musique_corpus.json"

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    title_to_docid = {}

    for doc_id, entry in corpus.items():
        title = entry["title"].strip()
        title_to_docid[title] = doc_id

    print("Loaded title → doc_id mappings:", len(title_to_docid))

    queries = {}
    qrels = []

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing_titles = set()

    for ex in data:
        qid = ex["_id"]
        queries[qid] = {
            "query": ex["question"]
        }

        for title, _ in ex["context"]:
            title = title.strip()
            if title in title_to_docid:
                doc_id = title_to_docid[title]
                qrels.append(f"{qid}\t{doc_id}\t1")
            else:
                missing_titles.add(title)

    # Write queries.train.json
    with open(f"{OUT_DIR}/queries.train.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    # Write qrels.train.tsv
    with open(f"{OUT_DIR}/qrels.train.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(qrels))

    print(f"Queries written: {len(queries)}")
    print(f"Qrels written:   {len(qrels)}")
    print(f"Missing titles: {len(missing_titles)}")

    if missing_titles:
        print("Example missing titles:", list(missing_titles)[:10])


def prepare_drhard_passage_dataset(
        corpus_json_path: str,
        train_json_path: str,
        out_dir: str = "data/passage/dataset"
):
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)

    with open(corpus_json_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    title_to_docid = {
        entry["title"].strip(): int(doc_id)
        for doc_id, entry in corpus.items()
    }

    with open(os.path.join(out_dir, "collection.tsv"), "w", encoding="utf-8") as out:
        for doc_id, entry in corpus.items():
            text = f'{entry["title"]} [SEP] {entry["text"]}'
            out.write(f"{int(doc_id)}\t{text}\n")

    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qid_map = {ex["_id"]: i for i, ex in enumerate(data)}

    with open(os.path.join(out_dir, "qid_map.json"), "w", encoding="utf-8") as f:
        json.dump(qid_map, f, indent=2)


    with open(os.path.join(out_dir, "queries.train.tsv"), "w", encoding="utf-8") as out:
        for ex in data:
            qid = qid_map[ex["_id"]]
            out.write(f"{qid}\t{ex['question']}\n")


    with open(os.path.join(out_dir, "qrels.train.tsv"), "w", encoding="utf-8") as out:
        for ex in data:
            qid = qid_map[ex["_id"]]
            for title, _ in ex["context"]:
                title = title.strip()
                if title in title_to_docid:
                    doc_id = title_to_docid[title]
                    out.write(f"{qid}\t0\t{doc_id}\t1\n")

    print("DRHard passage dataset prepared successfully.")
    print(f"Files written to: {out_dir}")

import json
import os

def rebuild_collection_tsv(
    corpus_json_path: str,
    out_path: str = "data/passage/dataset/collection.tsv"
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(corpus_json_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    bad = 0
    total = 0

    with open(out_path, "w", encoding="utf-8", newline="") as out:
        for doc_id, entry in corpus.items():
            try:
                pid = int(doc_id)
            except ValueError:
                bad += 1
                continue

            title = entry.get("title", "").replace("\n", " ").replace("\t", " ").strip()
            text = entry.get("text", "").replace("\n", " ").replace("\t", " ").strip()

            if not text:
                bad += 1
                continue

            passage = f"{title} [SEP] {text}".strip()
            out.write(f"{pid}\t{passage}\n")
            total += 1

    print(f"Done. Wrote {total} passages. Skipped {bad} bad entries.")


def sanity_check_collection_tsv(path="data/passage/dataset/collection.tsv", max_report=5):
    bad = 0
    total = 0
    bad_examples = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            total += 1
            line = line.rstrip("\n")

            if "\t" not in line:
                bad += 1
                bad_examples.append((lineno, "NO_TAB", line))
                continue

            pid, text = line.split("\t", 1)

            try:
                int(pid)
            except ValueError:
                bad += 1
                bad_examples.append((lineno, "NON_INT_ID", line))
                continue

            if not text.strip():
                bad += 1
                bad_examples.append((lineno, "EMPTY_TEXT", line))
                continue

    print(f"Checked {total} lines")
    print(f"Bad lines: {bad}")

    if bad_examples:
        print("\nFirst problematic lines:")
        for ex in bad_examples[:max_report]:
            lineno, reason, content = ex
            print(f"[Line {lineno}] {reason}: {content[:120]}")

    if bad == 0:
        print("\n collection.tsv is CLEAN and SAFE for DRHard preprocessing")
    else:
        print("\n collection.tsv still has issues — must fix before preprocessing")


def sanity_check_processing_embeddings():
    COLLECTION_PATH = "data/passage/dataset/collection.tsv"
    PASSAGE_MEMMAP = "data/passage/evaluate/contriever/passages.memmap"
    EMBED_DIM = 768  # contriever / roberta-base

    print("=== DRHard Passage Sanity Check ===")

    # ------------------------------------------------
    # 1) Check collection.tsv exists and count lines
    # ------------------------------------------------
    assert os.path.exists(COLLECTION_PATH), f"Missing: {COLLECTION_PATH}"

    with open(COLLECTION_PATH, "r", encoding="utf-8") as f:
        collection_lines = sum(1 for _ in f)

    print(f"[OK] collection.tsv lines        : {collection_lines}")

    # ------------------------------------------------
    # 2) Load passage.memmap safely
    # ------------------------------------------------
    assert os.path.exists(PASSAGE_MEMMAP), f"Missing: {PASSAGE_MEMMAP}"

    file_size = os.path.getsize(PASSAGE_MEMMAP)
    assert file_size % (4 * EMBED_DIM) == 0, \
        "passages.memmap size is not divisible by embedding dim"

    num_passages = file_size // (4 * EMBED_DIM)

    embeddings = np.memmap(
        PASSAGE_MEMMAP,
        dtype=np.float32,
        mode="r",
        shape=(num_passages, EMBED_DIM),
    )

    print(f"[OK] passage.memmap vectors     : {num_passages}")
    print(f"[OK] embedding dimension        : {EMBED_DIM}")


    if num_passages != collection_lines:
        print("\n MISMATCH DETECTED")
        print(f"collection.tsv lines : {collection_lines}")
        print(f"passage.memmap rows  : {num_passages}")
        print("\nThis WILL crash FAISS / CUDA during training.")
        raise SystemExit(1)

    print("[OK] collection ↔ memmap alignment")


    sample = embeddings[:1000]  # fast check
    if not np.isfinite(sample).all():
        print("\n Found NaN or Inf in passage embeddings")
        raise SystemExit(1)

    print("[OK] no NaN / Inf in embeddings")


    norms = np.linalg.norm(sample, axis=1)
    print(f"[INFO] embedding norm range     : {norms.min():.4f} → {norms.max():.4f}")

    print("\n SANITY CHECK PASSED")
    print("You are safe to run DRHard training.")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # do not run this, use existing memmap
    # main()
    # setup_training_files()
    # prepare_drhard_passage_dataset(
    #     corpus_json_path="data/wiki_musique_corpus.json",
    #     train_json_path="data/train.json",
    #     out_dir="data/passage/dataset"
    # )

    # rebuild_collection_tsv("data/wiki_musique_corpus.json")\
    # sanity_check_collection_tsv()
    # sanity_check_processing_embeddings()
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = RobertaDot.from_pretrained(
        "data/passage/adore/models/epoch-2"
    ).to("cuda")
    # model.eval()


