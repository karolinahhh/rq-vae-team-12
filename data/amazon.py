import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch
from datasets import load_dataset, load_from_disk

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable, List, Optional, Dict, Union

from PIL import Image
from torchvision import transforms


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)


class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str,  # 'beauty', 'sports', 'toys'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        category="brand",
    ) -> None:
        self.split = split
        self.brand_mapping = {}  # Dictionary to store brand_id -> brand_name mapping
        self.category = category
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]

    @property
    def processed_file_names(self) -> str:
        return f"data_{self.split}.pt"

    def download(self) -> None:
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, "data")
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def _remap_ids(self, x):
        return x - 1

    def get_brand_name(self, brand_id: int) -> str:
        """
        Returns the brand name for a given brand ID.

        Args:
            brand_id: The ID of the brand to look up

        Returns:
            The brand name as a string, or "Unknown" if the brand ID is not found
        """
        return self.brand_mapping.get(brand_id, "Unknown")

    def get_brand_mapping(self) -> Dict[int, str]:
        """
        Returns the complete brand ID to brand name mapping.

        Returns:
            Dictionary mapping brand IDs to brand names
        """
        return self.brand_mapping

    def train_test_split(self, max_seq_len=20):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []
        with open(
            os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r"
        ) as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]

                # We keep the whole sequence without padding. Allows flexible training-time subsampling.
                train_items = items[:-2]
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-2])

                eval_items = items[-(max_seq_len + 2) : -2]
                sequences["eval"]["itemId"].append(
                    eval_items + [-1] * (max_seq_len - len(eval_items))
                )
                sequences["eval"]["itemId_fut"].append(items[-2])

                test_items = items[-(max_seq_len + 1) : -1]
                sequences["test"]["itemId"].append(
                    test_items + [-1] * (max_seq_len - len(test_items))
                )
                sequences["test"]["itemId_fut"].append(items[-1])

        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])
        return sequences

    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), "r") as f:
            data_maps = json.load(f)

        # Construct user sequences
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"]) for k, v in sequences.items()
        }

        # Compute item features
        asin2id = pd.DataFrame(
            [
                {"asin": k, "id": self._remap_ids(int(v))}
                for k, v in data_maps["item2id"].items()
            ]
        )
        item_data = (
            pd.DataFrame(
                [
                    meta
                    for meta in parse(
                        path=os.path.join(self.raw_dir, self.split, "meta.json.gz")
                    )
                ]
            )
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": "Unknown"})
        )

        # Create brand mapping
        unique_brands = item_data[self.category].unique()
        self.brand_mapping = {i: brand for i, brand in enumerate(unique_brands)}

        # Create reverse mapping for lookup
        brand_to_id = {brand: i for i, brand in self.brand_mapping.items()}

        # Add brand_id to item_data
        item_data["brand_id"] = item_data["brand"].map(lambda x: brand_to_id.get(x, -1))

        sentences = item_data.apply(
            lambda row: "Title: "
            + str(row["title"])
            + "; "
            + "Brand: "
            + str(row["brand"])
            + "; "
            + "Categories: "
            + str(row["categories"][0])
            + "; "
            + "Price: "
            + str(row["price"])
            + "; ",
            axis=1,
        )

        # Store brand_id instead of brand name
        brand_ids = item_data.apply(lambda row: row["brand_id"], axis=1)

        item_emb = self._encode_text_feature(sentences)
        data["item"].x = item_emb
        data["item"].text = np.array(sentences)
        data["item"].brand_id = np.array(
            brand_ids
        )  # Store brand_id instead of brand name

        # Save the brand mapping to the data object as well
        data["brand_mapping"] = self.brand_mapping

        gen = torch.Generator()
        gen.manual_seed(42)
        data["item"].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        ########## Add train/val/test item splits ##########

        num_items = data["item"].x.shape[0]
        is_train = torch.zeros(num_items, dtype=torch.bool)
        is_val = torch.zeros(num_items, dtype=torch.bool)
        is_test = torch.zeros(num_items, dtype=torch.bool)

        # Create a deterministic random permutation of item indices
        gen = torch.Generator().manual_seed(42)
        perm = torch.randperm(num_items, generator=gen)

        n_total = num_items
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)

        train_ids = perm[:n_train]
        val_ids = perm[n_train:n_train + n_val]
        test_ids = perm[n_train + n_val:]

        is_train[train_ids] = True
        is_val[val_ids] = True
        is_test[test_ids] = True

        # data["item"].is_train = is_train
        # data["item"].is_val = is_val
        # data["item"].is_test = is_test
        data["item"]["is_train"] = is_train
        data["item"]["is_val"] = is_val
        data["item"]["is_test"] = is_test

        print("is_val in item:", "is_val" in data["item"])
        ##########

        self.save([data], self.processed_paths[0])

        # Save brand mapping to a separate file for easy access
        brand_mapping_path = os.path.join(
            self.processed_dir, f"brand_mapping_{self.split}.json"
        )
        with open(brand_mapping_path, "w") as f:
            json.dump(self.brand_mapping, f)

class AmazonReviews23(InMemoryDataset, PreprocessingMixin):

    def __init__(
        self,
        root: str,
        split: str, # "All_Beauty", "All_Sports", etc
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        category="store",
    ):
        
        self.split = split
        self.config_name = [f"raw_review_All_{split.capitalize()}", f"raw_meta_All_{split.capitalize()}"]
        self.category = category
        self.store_mapping = {}
        
        super(AmazonReviews23, self).__init__(root, transform, pre_transform, force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [f"{config}/state.json" for config in self.config_name]

    @property
    def processed_file_names(self) -> str:
        return f"data_{self.split}.pt"

    def download(self) -> None:
        for config in self.config_name:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                config,
                split="full",
                trust_remote_code=True,
            )
            ds.save_to_disk(osp.join(self.raw_dir, config))

    def _remap_ids(self, x):
        return x - 1

    def first_large(self, img_dict):
        urls = img_dict.get("large", np.array([], dtype=object))
        if isinstance(urls, np.ndarray) and urls.size > 0:
            return urls[0]
        return None

    def process_image(self, image_path):
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        image = img_transform(image)
        return image

    def process(self) -> None:
        # Reload the Arrow snapshot
        pdf_reviews = load_from_disk(osp.join(self.raw_dir, self.config_name[0])).to_pandas()
        pdf_items = load_from_disk(osp.join(self.raw_dir, self.config_name[1])).to_pandas()

        max_seq_len = 20
        user_ids = list(pdf_reviews["user_id"].unique())
        data_maps = {
            "item2id": {asin: idx for idx, asin in enumerate(pdf_reviews["parent_asin"].unique())},
            "user2id": {uid: idx for idx, uid in enumerate(user_ids)},
        }

        user_lists = {sp: [] for sp in ["train","eval","test"]}
        sequences = {sp: defaultdict(list) for sp in ["train","eval","test"]}
        for uid, grp in pdf_reviews.sort_values("timestamp").groupby("user_id", sort=False):

            raw_asins = grp["parent_asin"].tolist()
            if len(raw_asins) < 3:
                continue

            item_ids = [self._remap_ids(data_maps["item2id"][asin]) for asin in raw_asins]

            train = item_ids[:-2]
            eval_ = item_ids[-(max_seq_len+2):-2]
            test = item_ids[-(max_seq_len+1):-1]
            
            sequences["train"]["itemId"].append(train)
            sequences["train"]["itemId_fut"].append(item_ids[-2])
            user_lists["train"].append(uid)

            sequences["eval"]["itemId"].append(eval_ + [-1]*(max_seq_len - len(eval_)))
            sequences["eval"]["itemId_fut"].append(item_ids[-2])
            user_lists["eval"].append(uid)

            sequences["test"]["itemId"].append(test + [-1]*(max_seq_len - len(test)))
            sequences["test"]["itemId_fut"].append(item_ids[-1])
            user_lists["test"].append(uid)

        for sp in sequences:
            sequences[sp]["userId"] = [data_maps["user2id"][uid] for uid in user_lists[sp]]
            sequences[sp] = pl.from_dict(sequences[sp])

        # pdf_items["first_image"] = pdf_items["images"].apply(self.first_large)

        asin2id = pd.DataFrame([
            {"parent_asin": k, "id": self._remap_ids(int(v))}
            for k, v in data_maps["item2id"].items()
        ])

        item_data = (
            pdf_items
            .merge(asin2id, on="parent_asin")
            .sort_values(by="id")
            .reset_index(drop=True)
            .fillna({self.category: "Unknown"})
        )

        unique_stores = item_data[self.category].unique().tolist()
        self.store_mapping = {i: b for i, b in enumerate(unique_stores)}
        store_to_id = {b: i for i, b in self.store_mapping.items()}
        item_data["store_id"] = item_data[self.category].map(lambda x: store_to_id.get(x, -1))

        sentences = (
            "Title: " + item_data["title"].astype(str) + "; " +
            "Store: " + item_data["store"].astype(str) + "; " +
            "Categories: " + item_data["categories"].str[0].astype(str) + "; " +
            "Price: " + item_data["price"].astype(str) + "; "
        )
        # images = item_data["first_image"]

        item_emb = self._encode_text_feature(sentences)
        # item_emb_image = self._encode_image_feature([self.process_image(img) for img in images])

        data = HeteroData()
        data["user","rated","item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items()
        }

        data["item"].x = item_emb
        data["item"].text = np.array(sentences, dtype=object)
        data["item"].store_id = torch.tensor(item_data["store_id"].values)
        #data["item"].image = torch.tensor(item_emb_image)
        data["store_mapping"] = self.store_mapping

        gen = torch.Generator().manual_seed(42)
        data["item"].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        num_items = data["item"].x.shape[0]
        is_train = torch.zeros(num_items, dtype=torch.bool)
        is_val = torch.zeros(num_items, dtype=torch.bool)
        is_test = torch.zeros(num_items, dtype=torch.bool)

        n_total = num_items
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)

        perm = torch.randperm(num_items, generator=gen)

        train_ids = perm[:n_train]
        val_ids = perm[n_train:n_train + n_val]
        test_ids = perm[n_train + n_val:]

        is_train[train_ids] = True
        is_val[val_ids] = True
        is_test[test_ids] = True

        data["item"]["is_train"] = is_train
        data["item"]["is_val"] = is_val
        data["item"]["is_test"] = is_test

        # Save the processed graph object
        self.save([data], self.processed_paths[0])

        os.makedirs(self.processed_dir, exist_ok=True)
        with open(osp.join(self.processed_dir, f"store_mapping_{self.split}.json"), "w") as f:
            json.dump(self.store_mapping, f)