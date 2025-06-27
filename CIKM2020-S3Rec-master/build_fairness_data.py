import json
import pickle
import numpy as np
import argparse
from collections import defaultdict
from utils import get_user_seqs

def build_item_to_category_mapping(item2attribute_file, output_file):
    with open(item2attribute_file, 'r') as f:
        item2attribute = json.load(f)
    
    item_to_category = {}
    for item_id, attributes in item2attribute.items():
        if attributes:
            item_to_category[int(item_id)] = attributes[0]
        else:
            item_to_category[int(item_id)] = "UNKNOWN"
    
    with open(output_file, 'wb') as f:
        pickle.dump(item_to_category, f)
    
    print(f"Item to category mapping saved to {output_file}")
    print(f"Total items mapped: {len(item_to_category)}")
    return item_to_category

def build_user_category_profiles(data_file, item_to_category, output_file):
    user_seq, max_item, _, _ = get_user_seqs(data_file)
    
    user_category_profiles = {}
    
    for user_id, items in enumerate(user_seq):
        category_counts = defaultdict(int)
        
        for item in items[:-1]:
            category = item_to_category.get(item, "UNKNOWN")
            category_counts[category] += 1
        
        total_items = sum(category_counts.values())
        if total_items > 0:
            user_category_profiles[user_id] = {
                cat: count / total_items 
                for cat, count in category_counts.items()
            }
        else:
            user_category_profiles[user_id] = {}
    
    with open(output_file, 'wb') as f:
        pickle.dump(user_category_profiles, f)
    
    print(f"User category profiles saved to {output_file}")
    print(f"Total users profiled: {len(user_category_profiles)}")
    return user_category_profiles

def build_item_popularity(data_file, output_file):
    user_seq, max_item, _, _ = get_user_seqs(data_file)
    
    item_counts = defaultdict(int)
    total_interactions = 0
    
    for items in user_seq:
        for item in items:
            item_counts[item] += 1
            total_interactions += 1
    
    item_popularity = {
        item: count / total_interactions 
        for item, count in item_counts.items()
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(item_popularity, f)
    
    print(f"Item popularity saved to {output_file}")
    print(f"Total items: {len(item_popularity)}, Total interactions: {total_interactions}")
    return item_popularity

def build_user_popularity_profiles(data_file, item_popularity, output_file):

    user_seq, max_item, _, _ = get_user_seqs(data_file)
    
    user_popularity_profiles = {}
    
    for user_id, items in enumerate(user_seq):
        historical_items = items[:-1]
        
        if len(historical_items) > 0:
            popularity_scores = [item_popularity.get(item, 0) for item in historical_items]
            user_popularity_profiles[user_id] = np.mean(popularity_scores)
        else:
            user_popularity_profiles[user_id] = 0.0
    
    with open(output_file, 'wb') as f:
        pickle.dump(user_popularity_profiles, f)
    
    print(f"User popularity profiles saved to {output_file}")
    print(f"Total users profiled: {len(user_popularity_profiles)}")
    print(f"Average user popularity preference: {np.mean(list(user_popularity_profiles.values())):.6f}")
    return user_popularity_profiles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='Beauty', type=str, help='Dataset name (Beauty, Sports_and_Outdoors, Toys_and_Games)')
    parser.add_argument('--data_dir', default='data/', type=str, help='Data directory')
    args = parser.parse_args()
    
    data_file = f"{args.data_dir}{args.data_name}.txt"
    item2attribute_file = f"{args.data_dir}{args.data_name}_item2attributes.json"
    
    print(f"processing dataset: {args.data_name}")
    print(f"data file: {data_file}")
    print(f"item2attribute file: {item2attribute_file}")
    
    item_to_category = build_item_to_category_mapping(
        item2attribute_file,
        f"{args.data_dir}{args.data_name}_item_to_category.pkl"
    )
    
    user_profiles = build_user_category_profiles(
        data_file,
        item_to_category,
        f"{args.data_dir}{args.data_name}_user_category_profiles.pkl"
    )
    
    item_popularity = build_item_popularity(
        data_file,
        f"{args.data_dir}{args.data_name}_item_popularity.pkl"
    )
    
    user_popularity_profiles = build_user_popularity_profiles(
        data_file,
        item_popularity,
        f"{args.data_dir}{args.data_name}_user_popularity_profiles.pkl"
    )
    
    print("fairness data files created successfully")

if __name__ == "__main__":
    main()