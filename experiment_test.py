import json
from datetime import datetime
from generative_model_setup import llm, answer_question


def process_single_entry(entry):
    """
    Given a dataset entry containing question, answer, and context,
    generate an answer and save the result in the RAG experiment format.
    """

    # Extract question & gold answer
    question = entry["question"]
    gold_answer = entry["answer"]

    # Build context from entry["context"] which is [(title, [paragraphs])]
    raw_contexts = entry["context"]
    contexts = [
        f"{title}: {' '.join(paragraphs)}"
        for title, paragraphs in raw_contexts
    ]
    context_str = "\n\n".join(contexts)

    predicted_answer = answer_question(question, context_str)

    result_entry = {
        "id": entry.get("id"),
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        # this is just an example for the context id of an oracle, ideally we would save in separate
        # files for the different experiments
        # the id and stuff for the contexts used will be the ones of the docs from the corpus when retrieving
        # type will also be changed to negative, relative, oracle depending on the experiment step
        # the k refers to the top-k relevance and then we have some other stuff for ratio and model for note keeping
        "contexts_used": [
            {"doc_id": idx, "type": "oracle", "text": ctx}
            for idx, ctx in enumerate(contexts)
        ],
        "retriever": "oracle_only",
        "k": len(contexts),
        "ratio": None,
        "model": "llama3.1:8b",
        "timestamp": datetime.now().isoformat()
    }
    return result_entry


# -----------------------------
# Example usage for one entry
# -----------------------------
if __name__ == "__main__":
    with open("restricted_data/1200devTestSet.json", "r") as f:
        dataset = json.load(f)

    result = process_single_entry(dataset[0])
    with open("experiment_results/single_example.json", "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"Processed and saved entry: {result['id']}")
    print(json.dumps(result, indent=2))
