import numpy as np 
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ir.edit_distance import top_k_edit_distance

# load the data
with open("./../tea_data.json", "r") as f: 
    tea_data = json.load(f)["data"]

def build_tfidf(max_features, stop_words, max_df=0.8, min_df=10, norm="l2"):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, stop_words=stop_words, norm=norm)
    return vectorizer
    
# data variables
n_feats = 5000
tea_to_index = { tea:index for index, tea in enumerate(d["tea_category"] for d in tea_data) }
index_to_tea = { v:k for k, v in tea_to_index.items() }

def get_recommendations(search_tea, k):
    tfidf_vec = build_tfidf(n_feats, "english") 
    doc_tfidf = tfidf_vec.fit_transform([d["about"] for d in tea_data]).toarray()

    if search_tea.title() in tea_to_index:
        tea_index = tea_to_index[search_tea.title()] 
    else: 
        search_tea = top_k_edit_distance(search_tea=search_tea, k=1)[0]
        tea_index = tea_to_index[search_tea]
    search_data = tea_data[tea_index]
    query_tfidf = tfidf_vec.transform([search_data["about"]])

    sims = cosine_similarity(query_tfidf, doc_tfidf).flatten()
    ranked_ids = (-sims).argsort()

    data = []
    for tea_id in ranked_ids[:k]: 
        data.append({
            "tea_category": tea_data[tea_id]["tea_category"], 
            "tea_type": tea_data[tea_id]["tea_type"], 
            "about": tea_data[tea_id]["about"]
        })

    return json.dumps(data), search_tea

