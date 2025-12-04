import json
import re
import string


def normalize(text: str) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - strip whitespace
    - remove punctuation
    """
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text


def exact_match(pred: str, gold: str) -> int:
    """
    Exact Match (EM):
    Returns 1 if normalized predicted answer matches the gold answer exactly,
    otherwise 0.
    """
    return int(normalize(pred) == normalize(gold))


def cover_exact_match(pred: str, gold: str) -> int:
    """
    Cover Exact Match (CEM):
    Returns 1 if the normalized gold answer appears anywhere inside
    the normalized predicted answer, otherwise 0.
    """
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)
    return int(gold_norm in pred_norm)


if __name__ == "__main__":

    with open("experiment_results/single_example.json", "r") as f:
        results = json.load(f)

    if isinstance(results, dict):
        results = [results]

    updated_results = []

    for entry in results:
        gen_answer = entry["predicted_answer"]
        gold_answer = entry["gold_answer"]

        em_score = exact_match(gen_answer, gold_answer)
        cem_score = cover_exact_match(gen_answer, gold_answer)

        entry["EM"] = em_score
        entry["CEM"] = cem_score

        print(json.dumps(entry, indent=2))
        updated_results.append(entry)

    # Save updated entry back to file (optional)
    with open("experiment_results/single_example_scored.json", "w") as f:
        json.dump(updated_results, f, indent=2)

    print("\nScores added and saved to 'experiment_results/single_example_scored.json'")
