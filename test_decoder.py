import os
import torch
import gin
import json
from accelerate import Accelerator
from torch.utils.data import DataLoader

from data.processed import ItemData, SeqData, RecDataset
from data.utils import batch_to
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.model import EncoderDecoderRetrievalModel
from evaluate.metrics import TopKAccumulator
from modules.utils import parse_config
from modules.utils import compute_user_history_popularity

@gin.configurable
def evaluate_decoder(
    # Paths
    pretrained_rqvae_path: str,
    pretrained_decoder_path: str,
    dataset_folder: str,
    save_dir_root: str = "out/eval/",
    # Data & batching
    dataset: RecDataset = RecDataset.AMAZON,
    dataset_split: str = "sports",
    category: str = None,
    batch_size: int = 256,
    # VQ-VAE tokenizer
    vae_input_dim: int = 768,
    vae_hidden_dims: list = [512, 256, 128],
    vae_embed_dim: int = 32,
    vae_codebook_size: int = 256,
    vae_n_cat_feats: int = 0,
    vae_n_layers: int = 3,
    vae_codebook_normalize: bool = False,
    vae_sim_vq: bool = False,
    # Decoder
    decoder_embed_dim: int = 128,
    dropout_p: float = 0.3,
    attn_heads: int = 8,
    attn_embed_dim: int = 512,
    attn_layers: int = 8,
    model_jagged_mode: bool = True,
    # Inference
    ks: list = [1, 5, 10],
    temperature: float = 1.0,
    strong_generalization=False,
    reduce_users=False,
    split_dataset=False,
    split_qty=50,
):
    """
    Load a trained decoder and run evaluation on the held‐out split,
    printing out Hits@k, NDCG@k, Recall@k, and Gini@k.
    """
    accelerator = Accelerator()
    device = accelerator.device

    if dataset not in (RecDataset.AMAZON, RecDataset.AMAZON23):
        raise ValueError(f"Only AMAZON dataset supported, got {dataset}")
    
    amazon23_kwargs = {}
    if dataset == RecDataset.AMAZON23:
        amazon23_kwargs = {
            "split_dataset": split_dataset,
            "split_qty": split_qty,
        }
    
    item_dataset = (
        ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=False,
            split=dataset_split,
            category=category,
            strong_generalization=strong_generalization,
            reduce_users=reduce_users,
            **amazon23_kwargs,
        )
    )

    train_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        split_type="train",
        subsample=True,
        split=dataset_split,
        strong_generalization=strong_generalization,
        reduce_users=reduce_users,
        **amazon23_kwargs,
    )
    eval_seq = SeqData(
        root=dataset_folder,
        dataset=dataset,
        split_type="eval",
        subsample=False,
        split=dataset_split,
        strong_generalization=strong_generalization,
        reduce_users=reduce_users,
        **amazon23_kwargs,
    )
    test_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        split_type="test",
        subsample=False,
        split=dataset_split,
        strong_generalization=strong_generalization,
        reduce_users=reduce_users,
        **amazon23_kwargs,
    )
    eval_loader = DataLoader(eval_seq, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    device = next(tokenizer.parameters()).device
    user_gap_p_dict, global_popularity_dict, user_brand_dists_dict = compute_user_history_popularity(train_dataset, tokenizer, device)

    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=eval_seq.max_seq_len * tokenizer.sem_ids_dim,
        jagged_mode=model_jagged_mode,
    )
    # Load checkpoint
    ckpt = torch.load(pretrained_decoder_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = accelerator.prepare(model)

    eval_metrics_acc = TopKAccumulator(ks=ks, popularity_dict=global_popularity_dict, user_gap_p=user_gap_p_dict, user_brand_dists=user_brand_dists_dict)
    test_metrics_acc = TopKAccumulator(ks=ks, popularity_dict=global_popularity_dict, user_gap_p=user_gap_p_dict, user_brand_dists=user_brand_dists_dict)

    model.eval()
    model.enable_generation = True

    for batch in eval_loader:
        data = batch_to(batch, device)
        tokenized = tokenizer(data)
        with torch.no_grad():
            out = model.generate_next_sem_id(
                tokenized, top_k=True, temperature=temperature
            )
        actual = tokenized.sem_ids_fut        
        top_k = out.sem_ids
        user_ids = data.user_ids
        eval_metrics_acc.accumulate(actual=actual, top_k=top_k, user_ids=user_ids, tokenizer=tokenizer)

    for batch in test_loader:
        data = batch_to(batch, device)
        tokenized = tokenizer(data)
        with torch.no_grad():
            out = model.generate_next_sem_id(
                tokenized, top_k=True, temperature=temperature
            )
        actual = tokenized.sem_ids_fut        
        top_k = out.sem_ids
        user_ids = data.user_ids
        test_metrics_acc.accumulate(actual=actual, top_k=top_k, user_ids=user_ids, tokenizer=tokenizer)

    eval_results = eval_metrics_acc.reduce()
    test_results = test_metrics_acc.reduce()
    
    print("\n=== Evaluation Results ===")
    for metric, value in sorted(eval_results.items()):
        print(f"{metric:12s}: {value:.4f}")
    print("\n=== Test Results ===")
    for metric, value in sorted(test_results.items()):
        print(f"{metric:12s}: {value:.4f}")

    full_results = {
        "eval": eval_results,
        "test": test_results,
    }

    os.makedirs(save_dir_root, exist_ok=True)
    json_path = os.path.join(save_dir_root, "eval_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)


if __name__ == "__main__":
    parse_config()
    evaluate_decoder()