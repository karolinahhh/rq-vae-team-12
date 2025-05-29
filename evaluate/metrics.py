from collections import defaultdict
from torch import Tensor
import torch
import math
from einops import rearrange

def compute_dcg(relevance: list) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))

def compute_ndcg_for_semantic_ids(pred: Tensor, actual: Tensor, k: int) -> float:
    """
    Compute NDCG@k for one example of semantic ID tuples.
    pred: [K, D] tensor — top-k predicted semantic IDs
    actual: [D] tensor — ground truth semantic ID
    """
    actual_tuple = tuple(actual.tolist())  # Convert to hashable tuple
    relevance = [1 if tuple(row.tolist()) == actual_tuple else 0 for row in pred[:k]]
    dcg = compute_dcg(relevance)
    idcg = compute_dcg(sorted(relevance, reverse=True))
    return dcg / idcg if idcg > 0 else 0.0

class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        B, D = actual.shape
        pos_match = (rearrange(actual, "b d -> b 1 d") == top_k)
        for i in range(D):
            match_found, rank = pos_match[...,:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(matched_rank[matched_rank < k])
            
            match_found, rank = pos_match[...,i:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])

        B = actual.size(0)
        for b in range(B):
            gold_docs = actual[b]
            pred_docs = top_k[b]
            for k in self.ks:
                topk_pred = pred_docs[:k]
                hits = sum(1 for doc in topk_pred if doc in gold_docs)
                self.metrics[f"h@{k}"] += float(hits > 0)
                self.metrics[f"ndcg@{k}"] += compute_ndcg_for_semantic_ids(pred_docs, gold_docs, k)
        self.total += B

    def reduce(self) -> dict:
        return {k: v / self.total for k, v in self.metrics.items()}
