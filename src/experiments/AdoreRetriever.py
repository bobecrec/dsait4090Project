import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from transformers import AutoTokenizer
from dexter.retriever.BaseRetriever import BaseRetriver
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.evidence import Evidence
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity

# IMPORTANT: use RobertaDot, not AutoModel
from DRhard.adore.model import RobertaDot   # path inside DRHard/adore


class AdoreRetriever(BaseRetriver):
    """
    Dexter-compatible retriever using a trained ADORE query encoder
    and fixed passage embeddings.
    """

    def __init__(
        self,
        model_path: str,
        passage_memmap_path: str,
        batch_size: int = 32,
        device: str = "cuda"
    ):
        super().__init__()

        self.device = torch.device(device)
        self.batch_size = batch_size

        # Tokenizer must match training
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

        # Load trained ADORE query encoder
        self.model = RobertaDot.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load passage embeddings (already normalized)
        self.passage_embeddings = np.memmap(
            passage_memmap_path,
            dtype=np.float32,
            mode="r"
        ).reshape(-1, self.model.output_embedding_size)

        self.sim = CosineSimilarity()

    @torch.no_grad()
    def encode_queries(self, queries: List[Question]) -> torch.Tensor:
        texts = [q.text() for q in queries]
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # ADORE query encoding
            emb = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                is_query=True
            )

            # Normalize to match passage space
            emb = torch.nn.functional.normalize(emb, dim=1)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    def retrieve(
        self,
        corpus: List[Evidence],
        queries: List[Question],
        top_k: int,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:

        query_embeddings = self.encode_queries(queries)
        passage_embeddings = torch.from_numpy(self.passage_embeddings)

        scores = self.sim.evaluate(query_embeddings, passage_embeddings)
        topk_scores, topk_idx = torch.topk(scores, top_k, dim=1)

        corpus_ids = [doc.id() for doc in corpus]
        results = {}

        for qi, q in enumerate(queries):
            results[q.id()] = {
                corpus_ids[int(pid)]: float(topk_scores[qi, j])
                for j, pid in enumerate(topk_idx[qi])
            }

        return results
