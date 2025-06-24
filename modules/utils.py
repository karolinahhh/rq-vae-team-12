import argparse
import gin
import torch
from data.schemas import TokenizedSeqBatch
from einops import rearrange
from torch import Tensor
from collections import defaultdict, Counter
import numpy as np


def reset_kv_cache(fn):
    def inner(self, *args, **kwargs):
        self.decoder.reset_kv_cache()
        out = fn(self, *args, **kwargs)
        self.decoder.reset_kv_cache()
        return out

    return inner


def reset_encoder_cache(fn):
    def inner(self, *args, **kwargs):
        if self.jagged_mode:
            self.transformer.cached_enc_output = None
        out = fn(self, *args, **kwargs)
        if self.jagged_mode:
            self.transformer.cached_enc_output = None
        return out

    return inner


def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def select_columns_per_row(x: Tensor, indices: Tensor) -> torch.Tensor:
    assert x.shape[0] == indices.shape[0]
    assert indices.shape[1] <= x.shape[1]

    B = x.shape[0]
    return x[rearrange(torch.arange(B, device=x.device), "B -> B 1"), indices]


def maybe_repeat_interleave(x, repeats, dim):
    if not isinstance(x, Tensor):
        return x
    return x.repeat_interleave(repeats, dim=dim)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to gin config file.")
    args = parser.parse_args()
    gin.parse_config_file(args.config_path)


@torch.no_grad
def compute_debug_metrics(
    batch: TokenizedSeqBatch, model_output=None, prefix: str = ""
) -> dict:
    seq_lengths = batch.seq_mask.sum(axis=1).to(torch.float32)
    prefix = prefix + "_"
    debug_metrics = {
        prefix
        + f"seq_length_p{q}": torch.quantile(seq_lengths, q=q).detach().cpu().item()
        for q in [0.25, 0.5, 0.75, 0.9, 1]
    }
    if model_output is not None:
        loss_debug_metrics = {
            prefix + f"loss_{d}": model_output.loss_d[d].detach().cpu().item()
            for d in range(batch.sem_ids_fut.shape[1])
        }
        debug_metrics.update(loss_debug_metrics)
    return debug_metrics


@torch.no_grad
@torch._dynamo.disable
def compute_user_history_popularity(train_dataset, tokenizer, device):
    user_pop = defaultdict(list)
    global_pop = Counter()
    user_brand_counts = defaultdict(Counter)

    for sample in train_dataset:
        user_id = sample.user_ids.item()  # assumes batch size 1
        ids = sample.ids[sample.ids >= 0]  # ignore padding (-1s)
        for item_id in ids:
            # tokenize that single item into semantic ID
            sem_id = tokenizer.cached_ids[item_id].to(device)
            # convert to string for search purposes
            key = str(sem_id.tolist())
            # track popularity
            user_pop[user_id].append(key)
            global_pop[key] += 1
            # track brand 
            brand = tokenizer.map_to_category.get(key, "unknown")
            user_brand_counts[user_id][brand] += 1

    # Map user_id → average historical item popularity
    user_gap_p = {}
    for user, items in user_pop.items():
        popularities = [global_pop[item] for item in items]
        user_gap_p[user] = np.mean(popularities)
    #brand distributions
    user_brand_dists = {
        user: {b: count / sum(brand_counts.values()) for b, count in brand_counts.items()}
        for user, brand_counts in user_brand_counts.items()
    }

    return user_gap_p, dict(global_pop), user_brand_dists