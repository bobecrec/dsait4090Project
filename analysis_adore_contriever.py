import json
import pandas as pd


NO_INFO_PATTERNS = [
    "not enough information",
    "do not have enough",
    "cannot determine",
    "cannot answer",
    "insufficient information",
    "unknown",
    "do not mention",
    "no information",
    "any information",
    "enough information",
    "not contain",
    "not provide",
    # explicit uncertainty
    "i do not have enough information",
    "i don't have enough information",
    "i cannot determine",
    "i cannot answer",
    "i am unable to determine",
    "i could not find any information",
    "i cannot provide a conclusive answer",

    # document/context missing
    "the provided document does not contain",
    "the provided documents do not contain",
    "the documents do not contain",
    "the documents provided do not contain",
    "the given document does not contain",

    # relevance mismatch
    "the question does not appear to be related",
    "not relevant to answering",
    "this information is not relevant",
    "document is about a different",

    # absence statements
    "there is no information provided about",
    "there is no information about",
    "the documents do not mention",

    # short abstentions
    "unknown",
    "n/a",
]

def adore_topk_performance_table(ks=(1, 3, 5)):
    rows = []

    for k in ks:
        path = f"experiment_results/adore/adore_top{k}_with_gold_scored.json"
        with open(path, "r") as f:
            data = json.load(f)

        results = data["results"]
        n = len(results)

        exact = sum(r["exact_match"] for r in results)
        cover_match = sum(r["cover_exact_match"] == 1 for r in results)
        cover_only = sum(
            (r["cover_exact_match"] == 1 and r["exact_match"] == 0)
            for r in results
        )

        none = n - exact - cover_only

        no_info = sum(
            any(pat in r["prediction"].lower() for pat in NO_INFO_PATTERNS)
            for r in results
        )

        rows.append({
            "k": k,
            "Exact Match (%)": 100 * exact / n,
            "Cover Exact Match (%)": 100 * cover_match/n,
            "None (%)": 100 * none / n,
            "No-information (%)": 100 * no_info / n,
        })

    return pd.DataFrame(rows).set_index("k")


if __name__ == "__main__":
    df = adore_topk_performance_table()
    print(df.round(2))
