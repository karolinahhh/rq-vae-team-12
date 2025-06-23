from collections import defaultdict
from torch import Tensor
import torch
import math
from einops import rearrange
import numpy as np


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


class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        # count frequencies
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        return self.gini_coefficient(list(freqs.values()))


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10], popularity_dict=None, user_gap_p=None, user_brand_dists=None):
        self.ks = ks
        self.popularity_dict = popularity_dict or {}
        self.user_gap_p = user_gap_p or {}
        self.user_brand_dists = user_brand_dists or {}
        self.coverage_sets = defaultdict(set)
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)

    def accumulate(self, actual: Tensor, top_k: Tensor, user_ids: Tensor, tokenizer=None) -> None:
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k
        for i in range(D):
            match_found, rank = pos_match[..., : i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(
                    matched_rank[matched_rank < k]
                )

            match_found, rank = pos_match[..., i : i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])

        B = actual.size(0)
        for b in range(B):
            gold_docs = actual[b]
            pred_docs = top_k[b]
            user_id = user_ids[b].item() ####
            for k in self.ks:
                topk_pred = pred_docs[:k]
                # print(f"User {user_id} | pred_docs shape: {pred_docs.shape} | pred_docs: {pred_docs}")

                ########### Popularity-aware fairness (GAP)
                # print("topk_pred:", topk_pred)
                # print("Pred key:", str(topk_pred[0].tolist()))

                if self.popularity_dict and self.user_gap_p:
                    def get_popularity(semantic_id):
                        key = str(semantic_id.tolist())
                        return self.popularity_dict.get(key, 0)

                    user_p = self.user_gap_p.get(user_id, 1)  # avoid divide-by-zero

                    pred_pop = torch.tensor([get_popularity(pred) for pred in topk_pred], dtype=torch.float)
                    GAP_r = pred_pop.mean().item()

                    delta_gap_user = (GAP_r - user_p) / user_p
                    self.metrics[f"delta_gap_user@{k}"] += delta_gap_user

                ##############
                hits = torch.any(torch.all(topk_pred == gold_docs, dim=1)).item()
                self.metrics[f"h@{k}"] += float(hits > 0)
                self.metrics[f"ndcg@{k}"] += compute_ndcg_for_semantic_ids(
                    pred_docs, gold_docs, k
                )
                ######### Top-K coverage #############
                for pred in topk_pred:
                    pred_tuple = tuple(pred.tolist())  
                    self.coverage_sets[k].add(pred_tuple)
                # if the tokinzer is given then for each prediction find the catergoy and add it to the list and then caclulate the gini coefficient
                if tokenizer is not None:
                    ############ Gini coefficient over brands #############
                    list_gini = []
                    for pred in topk_pred:
                        # idx = str(pred.tolist()[:-1]) 
                        idx = str(pred.tolist()) #don't remove the deduplication token
                        category = tokenizer.map_to_category[idx]
                        list_gini.append({"id": idx, "category": category})
                    self.metrics[f"gini@{k}"] += GiniCoefficient().calculate_list_gini(
                        list_gini, key="category"
                    )
                    ############ KL divergence over brand distribution #############
                    if self.user_brand_dists:
                        # Get predicted brand counts
                        pred_brands = [
                            tokenizer.map_to_category.get(str(pred.tolist()), "UNKNOWN")
                            for pred in topk_pred
                        ]
                        pred_counts = defaultdict(int)
                        for b in pred_brands:
                            pred_counts[b] += 1
                        # print("pred_counts:", pred_counts) 

                        total_pred = sum(pred_counts.values())
                        pred_dist = {k: v / total_pred for k, v in pred_counts.items()}

                        # Get user historical brand distribution
                        hist_dist = self.user_brand_dists.get(user_id, {})
                        all_brands = set(hist_dist.keys()) | set(pred_dist.keys())

                        p = torch.tensor([hist_dist.get(b, 1e-8) for b in all_brands])
                        q = torch.tensor([pred_dist.get(b, 1e-8) for b in all_brands])

                        # KL divergence: D_KL(p || q)
                        kl_div = torch.sum(p * torch.log(p / q)).item()
                        # print("kl_div:", kl_div)
                        
                        self.metrics[f"kl_brand@{k}"] += kl_div
                
        self.total += B

    # def reduce(self) -> dict:
    #     return {k: v / self.total for k, v in self.metrics.items()}
    def reduce(self) -> dict:
        result = {k: v / self.total for k, v in self.metrics.items()}
        for k in self.ks:
            total_preds = self.total * k
            unique_preds = len(self.coverage_sets.get(k, []))
            result[f"coverage@{k}"] = unique_preds / total_preds if total_preds > 0 else 0.0
        return result