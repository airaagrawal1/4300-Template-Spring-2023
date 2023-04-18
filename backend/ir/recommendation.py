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
    """
    Returns a TfidfVectorizer object with the given parameters as its preprocessing properties.

    TfidfVectorizer object is a document-by-vocabulary matrix, where entry i,j 
    corresponds to the tfidf score of word j in document i.
    """
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                 max_features=max_features, stop_words=stop_words, norm=norm)
    return vectorizer


# data variables
n_feats = 5000
tea_to_index = {tea: index for index, tea in enumerate(
    d["tea_category"] for d in tea_data)}
index_to_tea = {v: k for k, v in tea_to_index.items()}

# get the query vector (determine if there is a close match or it is a free-form query)


def get_query(search_tea, about_vec, review_vec): 
    """
    Returns a matrix of the tfidf between the query and the tea abouts and a matrix of 
    the tfidf weights between the query and the tea reviews. It also returns the 
    top k teas most similar (based on edit distance) to the original tea query

    Parameters
    ----------
    search_tea : string 
        the tea type query 
    """
    top_k_teas, top_k_dists = top_k_edit_distance(search_tea=search_tea, k=5)

    percent_edit = top_k_dists[0] / len(search_tea)
    threshold = 0.5  # half of the original query had to be edited
    if percent_edit > threshold:  # the query likely does not match any tea
        about_query_tfidf = about_vec.transform([search_tea])
        review_query_tfidf = review_vec.transform([search_tea])
    else:
        tea_index = tea_to_index[top_k_teas[0]]
        search_data = tea_data[tea_index]
        about_query_tfidf = about_vec.transform([search_data["about"]])
        review_query_tfidf = review_vec.transform(
            [" ".join(search_data["reviews"])])

    return about_query_tfidf, review_query_tfidf, top_k_teas

#def get_order


def get_recommendations(search_tea, k):
    about_tfidf_vec = build_tfidf(n_feats, "english")
    review_tfidf_vec = build_tfidf(n_feats, "english")
    about_tfidf = about_tfidf_vec.fit_transform(
        [d["about"] for d in tea_data]).toarray()
    review_tfidf = review_tfidf_vec.fit_transform(
        [" ".join(d["reviews"]) for d in tea_data]).toarray()

    about_query_tfidf, review_query_tfidf, top_k_teas = get_query(
        search_tea.title(), about_tfidf_vec, review_tfidf_vec)

    about_sims = cosine_similarity(about_query_tfidf, about_tfidf).flatten()
    review_sims = cosine_similarity(review_query_tfidf, review_tfidf).flatten()

    about_weight = 0.6
    review_weight = 0.4
    merged_sims = about_weight * about_sims + review_weight * review_sims
    ranked_ids = (-merged_sims).argsort()

    data = []
    counter = 0 
    while (len(data) < 5):
        tea_id = ranked_ids[counter]
        if(tea_data[tea_id]["tea_category"] != search_tea):
            data.append({
                "tea_category": tea_data[tea_id]["tea_category"],
                "tea_type": tea_data[tea_id]["tea_type"],
                "about": tea_data[tea_id]["about"],
                "brands": tea_data[tea_id]["top_rated_brands"],
                "score": merged_sims[tea_id]
            })
        counter+=1

    result = {
        "data": data,
        "top_k_teas": top_k_teas
    }

    return json.dumps(result)
