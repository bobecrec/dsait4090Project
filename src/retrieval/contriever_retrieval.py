from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.retriever.dense.DenseFullSearch import DenseHyperParams
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity


def run_contriever(k_retrieval: int = 10):
    """
    Returns:
      queries: dict[qid -> text]
      qrels:   dict[qid -> {doc_id: relevance}]
      corpus:  dict[doc_id -> {title, text, ...}]
      results: dict[qid -> list[(doc_id, score)]]
    """
    loader = RetrieverDataset(
        "wikimultihopqa",         # dataset alias
        "wikimultihopqa-corpus",  # corpus alias
        "config.ini",
        Split.DEV,
        tokenizer=None,
    )

    queries, qrels, corpus = loader.qrels()

    cfg = DenseHyperParams(
        query_encoder_path="facebook/contriever",
        document_encoder_path="facebook/contriever",
        batch_size=4,
        show_progress_bar=True,
    )

    retriever = Contriever(cfg)

    sim = CosineSimilarity()

    results = retriever.retrieve(
        corpus,
        queries,
        k_retrieval,
        sim,
        chunk=True,
        chunksize=400_000,
    )

    return queries, qrels, corpus, results


if __name__ == "__main__":
    queries, qrels, corpus, results = run_contriever(k_retrieval=5)
    for i, (qid, qtext) in enumerate(queries.items()):
        if i >= 3:
            break
        print(f"\nQID: {qid}")
        print("Q:", qtext)
        for doc_id, score in results[qid][:3]:
            print(f"  DOC {doc_id} (score={score:.3f}) :: {corpus[doc_id]['title']}")


