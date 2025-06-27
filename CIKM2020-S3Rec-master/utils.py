# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 11:06
# @Author  : Hui Wang

import numpy as np
import math
import random
import os
import json
import pickle
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F

from collections import defaultdict

import numpy as np
import math
import torch
from collections import defaultdict

class GiniCoefficient:
    
    def gini_coefficient(self, values):
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            return 0.0
        
        arr = np.sort(arr)
        n = arr.size
        mu = arr.mean()
        if mu == 0:
            return 0.0
            
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        return gini

    def calculate_category_gini(self, item_ids, item_to_category):
        category_counts = defaultdict(int)
        for item_id in item_ids:
            category = item_to_category.get(item_id, "UNKNOWN")
            category_counts[category] += 1
        
        return self.gini_coefficient(list(category_counts.values()))

def compute_kl_divergence(p_dist, q_dist, all_categories=None):
    if all_categories is None:
        all_categories = set(p_dist.keys()) | set(q_dist.keys())
    
    kl_div = 0.0
    for category in all_categories:
        p = p_dist.get(category, 1e-8)
        q = q_dist.get(category, 1e-8)
        kl_div += p * math.log(p / q)
    
    return kl_div

def compute_coverage(all_predictions, total_possible_items=None):
    unique_items = len(set(all_predictions))
    if total_possible_items is None:
        return unique_items
    return unique_items / total_possible_items

class FairnessMetrics:
    """
    Compute fairness and diversity metrics for recommendation systems.
    """
    
    def __init__(self, item_to_category=None, user_category_profiles=None, 
                 item_popularity=None, user_popularity_profiles=None, total_items=None):
        self.item_to_category = item_to_category or {}
        self.user_category_profiles = user_category_profiles or {}
        self.item_popularity = item_popularity or {}
        self.user_popularity_profiles = user_popularity_profiles or {}  # NEW: user's historical avg popularity
        self.total_items = total_items
        self.gini_calc = GiniCoefficient()
        self.global_predicitons = defaultdict(set)  
        
    def compute_metrics_for_predictions(self, user_ids, pred_lists, ks=[1, 5, 10]):

        metrics = defaultdict(float)
        all_predictions = defaultdict(set)  # k -> set of predicted items
        
        batch_size = len(user_ids)
        
        
        for b in range(min(3, batch_size)):
            user_id = user_ids[b]
            pred_items = pred_lists[b]
            
            
            for k in ks:
                topk_items = pred_items[:k]
                
                # coverage tracking
                for item in topk_items:
                    all_predictions[k].add(item)
                
                # gini coefficient over categories
                if self.item_to_category:
                    gini = self.gini_calc.calculate_category_gini(
                        topk_items, self.item_to_category
                    )
                    metrics[f"gini@{k}"] += gini
                
                # KL divergence between user profile and recommendations
                if self.user_category_profiles and user_id in self.user_category_profiles:
                    # get predicted category distribution
                    pred_category_counts = defaultdict(int)
                    for item in topk_items:
                        category = self.item_to_category.get(item, "UNKNOWN")
                        pred_category_counts[category] += 1
                    
                    # normalize to probabilities
                    total_pred = sum(pred_category_counts.values())
                    if total_pred > 0:
                        pred_dist = {cat: count/total_pred for cat, count in pred_category_counts.items()}
                        
                        # get user's historical category distribution
                        user_dist = self.user_category_profiles[user_id]
                        
                        # compute KL divergence
                        kl_div = compute_kl_divergence(user_dist, pred_dist)
                        metrics[f"kl_divergence@{k}"] += kl_div
                
                # avg popularity of recommended items
                if self.item_popularity:
                    avg_popularity = np.mean([self.item_popularity.get(item, 0) for item in topk_items])
                    metrics[f"avg_popularity@{k}"] += avg_popularity
                    
                    # delta GAP: difference between recommended and user's historical popularity preference
                    if self.user_popularity_profiles and user_id in self.user_popularity_profiles:
                        user_avg_pop = self.user_popularity_profiles[user_id]
                        if user_avg_pop > 0:  # Avoid division by zero
                            delta_gap = (avg_popularity - user_avg_pop) / user_avg_pop
                            metrics[f"delta_gap@{k}"] += delta_gap
        
        # normalize by batch size
        for key in metrics:
            metrics[key] /= batch_size
        
        
        # add coverage metrics
        for k in ks:
            coverage = compute_coverage(all_predictions[k], self.total_items)
            metrics[f"old_overage@{k}"] = coverage
            
        return dict(metrics)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix

def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        long_sequence.extend(items) # 后面的都是采的负例
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq

def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set) # 331
    return item2attribute, attribute_size

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res