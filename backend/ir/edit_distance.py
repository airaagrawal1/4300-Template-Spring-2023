import json
import numpy as np
from nltk import edit_distance

# load the data
with open("./../tea_data.json", "r") as f: 
    tea_data = json.load(f)["data"]

# data variables
n_feats = 5000
tea_to_index = { tea:index for index, tea in enumerate(d["tea_category"] for d in tea_data) }
index_to_tea = { v:k for k, v in tea_to_index.items() }

def top_k_edit_distance(search_tea, k):
    """
    Returns a ranking of the top k teas in the database ranked by the levenshtein edit distance 
    from the search_tea

    Parameters
    ----------
    search_tea : string 
        the tea type query 
    k: int 
        the number of ranked teas to return
    """
    tea_names = [t["tea_category"] for t in tea_data]
    edit_distances = [edit_distance(search_tea, tea_name) for tea_name in tea_names]
    ranked_ids = np.argsort(edit_distances)
    sorted_distances = np.sort(edit_distances)
    return [index_to_tea[i] for i in ranked_ids][:k], sorted_distances[:k]