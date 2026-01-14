from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from src.experiments.AdoreRetriever import AdoreRetriever


def run_adore(k=50):
    loader = RetrieverDataset(
        "wikimultihopqa",
        "wikimultihopqa-corpus",
        "config.ini",
        Split.DEV,
        tokenizer=None,
    )

    queries, qrels, corpus = loader.qrels()

    retriever = AdoreRetriever(
        model_path="../data/passage/adore/models/epoch-2",
        passage_memmap_path="../data/passage/evaluate/contriever/passages.memmap",
        batch_size=32,
        device="cuda"
    )

    results = retriever.retrieve(
        corpus=list(corpus.values()),
        queries=list(queries.values()),
        top_k=k
    )

    return results


if __name__ == "__main__":
    results = run_adore(50)
    for qid in list(results.keys())[:1]:
        print(qid, list(results[qid].items())[:5])
