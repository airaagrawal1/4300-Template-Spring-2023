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
tea_to_index = {tea: index for index, tea in enumerate(
    d["tea_category"] for d in tea_data)}
index_to_tea = {v: k for k, v in tea_to_index.items()}

# create a tfidf vector
def create_vector(data, query): 
    tfidf_vectorizer = build_tfidf(n_feats, "english")
    data_tfidf = tfidf_vectorizer.fit_transform(data).toarray()
    query_tfidf = tfidf_vectorizer.transform(query).toarray()
    return data_tfidf, query_tfidf

# calculate cosine sims 
def cosine_sims(about_tfidf, review_tfidf, about_query_tfidf, review_query_tfidf): 
    about_sims = cosine_similarity(about_query_tfidf, about_tfidf).flatten()
    review_sims = cosine_similarity(review_query_tfidf, review_tfidf).flatten()
    return about_sims, review_sims

# get similarity scores
def get_sim_scores(search_tea, search_description, k):
    about_weight = 0.6
    review_weight = 0.4
    if search_tea: 
        top_k_teas, top_k_dists = top_k_edit_distance(search_tea=search_tea, k=k)
        percent_edit = top_k_dists[0] / len(search_tea)
        threshold = 0.5
        if percent_edit < threshold: # the query is likely a valid tea
            search_tea = top_k_teas[0]
            search_data = tea_data[tea_to_index[search_tea]]
            about_tfidf, about_query_tfidf = create_vector([d["about"] for d in tea_data], [search_data["about"]])
            review_tfidf, review_query_tfidf = create_vector([" ".join(d["reviews"]) for d in tea_data], [" ".join(search_data["reviews"])])
            about_sims, review_sims = cosine_sims(about_tfidf, review_tfidf, about_query_tfidf, review_query_tfidf)
            search_tea_merged_sims = about_weight * about_sims + review_weight * review_sims
            if not search_description: 
                return search_tea_merged_sims, [search_tea]
    if search_description: 
        about_tfidf, about_query_tfidf = create_vector([d["about"] for d in tea_data], [search_description])
        review_tfidf, review_query_tfidf = create_vector([" ".join(d["reviews"]) for d in tea_data], [search_description])
        about_sims, review_sims = cosine_sims(about_tfidf, review_tfidf, about_query_tfidf, review_query_tfidf)
        description_merged_sims = about_weight * about_sims + review_weight * review_sims
        if not search_tea: 
            return description_merged_sims, []
    
    search_weight = 0.5
    description_weight = 0.5

    return search_weight * search_tea_merged_sims + description_weight * description_merged_sims, [search_tea]

# get recommendations
def get_recommendations(search_tea, search_description, k):
    sim_scores, search_tea = get_sim_scores(search_tea=search_tea, search_description=search_description, k=k)
    ranked_ids = (-sim_scores).argsort()
    if search_tea != []: 
        ranked_ids = ranked_ids[ranked_ids != tea_to_index[search_tea[0]]] # remove the current search
        
    data = []
    for tea_id in ranked_ids[:k]:
        data.append({
            "tea_category": tea_data[tea_id]["tea_category"],
            "tea_type": tea_data[tea_id]["tea_type"],
            "about": tea_data[tea_id]["about"],
            "brands": tea_data[tea_id]["top_rated_brands"],
            "score": sim_scores[tea_id] 
        })

    result = {
        "data": data
    }

    return json.dumps(result)