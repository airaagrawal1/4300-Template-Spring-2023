import numpy as np 
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ir.edit_distance import top_k_edit_distance

# load the data
with open("./../tea_data.json", "r") as f: 
    tea_data = json.load(f)["data"]

# vectorizer
def build_tfidf(max_features, stop_words, max_df=0.8, min_df=10, norm="l2"):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, stop_words=stop_words, norm=norm)
    return vectorizer
    
# data variables
n_feats = 5000
tea_to_index = { tea:index for index, tea in enumerate(d["tea_category"] for d in tea_data) }
index_to_tea = { v:k for k, v in tea_to_index.items() }

def get_recommendations(search_tea, k):
    about_tfidf_vec = build_tfidf(n_feats, "english") 
    review_tfidf_vec = build_tfidf(n_feats, "english") 
    about_tfidf = about_tfidf_vec.fit_transform([d["about"] for d in tea_data]).toarray()
    review_tfidf = review_tfidf_vec.fit_transform([" ".join(d["reviews"]) for d in tea_data]).toarray()

    if not search_tea.title() in tea_to_index:
        top_k_teas = top_k_edit_distance(search_tea=search_tea, k=5)
        search_tea = top_k_teas[0]

    tea_index = tea_to_index[search_tea.title()] 
    search_data = tea_data[tea_index]
    about_query_tfidf = about_tfidf_vec.transform([search_data["about"]])
    review_query_tfidf = review_tfidf_vec.transform([" ".join(search_data["reviews"])])

    about_sims = cosine_similarity(about_query_tfidf, about_tfidf).flatten()
    review_sims = cosine_similarity(review_query_tfidf, review_tfidf).flatten()

    about_weight = 0.5
    review_weight = 0.5
    merged_sims = about_weight * about_sims + review_weight * review_sims
    ranked_ids = (-merged_sims).argsort()

    data = []
    for tea_id in ranked_ids[:k]: 
        data.append({
            "tea_category": tea_data[tea_id]["tea_category"], 
            "tea_type": tea_data[tea_id]["tea_type"], 
            "about": tea_data[tea_id]["about"]
        })

    result = {
        "data": data, 
        "search_tea": search_tea, 
        "top_k_teas": top_k_teas
    }

    return json.dumps(result)

