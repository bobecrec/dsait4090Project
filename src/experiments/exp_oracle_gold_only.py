"""
Experiment: Oracle Gold Only

Uses only the gold/annotated contexts from supporting_facts.
No retrieval involved - this establishes the upper bound performance.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from src.generative_model_setup import generate_answer


def load_dev_data(path: str, num_samples: int = 1200) -> List[Dict[str, Any]]:
    """Load dev.json and return first num_samples."""
    with open(path, "r") as f:
        data = json.load(f)
    return data[:num_samples]


def extract_oracle_contexts(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract gold contexts from a sample using supporting_facts.

    supporting_facts: [[title, sent_idx], ...]
    context: [[title, [sent0, sent1, ...]], ...]
    """
    # Build title -> sentences mapping
    title_to_sentences = {}
    for ctx in sample.get("context", []):
        if len(ctx) >= 2:
            title = ctx[0]
            sentences = ctx[1]
            title_to_sentences[title] = sentences

    # Extract supporting facts
    contexts = []
    seen = set()
    for fact in sample.get("supporting_facts", []):
        if len(fact) >= 2:
            title = fact[0]
            sent_idx = fact[1]

            if title in title_to_sentences:
                sentences = title_to_sentences[title]
                if sent_idx < len(sentences):
                    text = sentences[sent_idx]
                    key = (title, text)
                    if key not in seen:
                        contexts.append({
                            "doc_id": f"oracle_{title}_{sent_idx}",
                            "title": title,
                            "text": text,
                            "score": 1.0,  # Oracle contexts have perfect relevance
                        })
                        seen.add(key)

    return contexts


def run_oracle_gold_only(
    dev_path: str = "restricted_data/dev.json",
    out_path: str = "experiment_results/oracle_gold_only.json",
    num_samples: int = 1200,
) -> None:
    """
    Run oracle experiment using only gold contexts.

    This establishes the upper bound - what performance is achievable
    with perfect retrieval (only relevant documents).
    """
    print(f"[oracle_gold_only] Loading {num_samples} samples from {dev_path}...")
    data = load_dev_data(dev_path, num_samples)

    output: List[Dict[str, Any]] = []

    print(f"[oracle_gold_only] Processing {len(data)} questions with oracle contexts...")
    for i, sample in enumerate(data):
        qid = sample["_id"]
        question = sample["question"]
        gold_answer = sample.get("answer", "")

        # Extract oracle contexts
        contexts = extract_oracle_contexts(sample)

        # Generate answer
        prediction = generate_answer(question, contexts)

        output.append({
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "contexts": contexts,
            "prediction": prediction,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} questions...")

    # Save results
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[oracle_gold_only] Saved {len(output)} examples -> {out_path}")


if __name__ == "__main__":
    run_oracle_gold_only()
