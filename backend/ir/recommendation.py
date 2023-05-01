import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

# load the data
with open("./tea_data.json", "r") as f:
    tea_data = json.load(f)["data"]

# make term document matrix
vectorizer =  TfidfVectorizer(stop_words = "english", max_df = .8, norm="l2")
td_matrix = vectorizer.fit_transform([t["doc"] for t in tea_data])
docs_compressed, s, words_compressed = svds(td_matrix, k=40)
words_compressed = words_compressed.transpose()
docs_compressed_normed = normalize(docs_compressed)
words_compressed_normed = normalize(words_compressed, axis = 1)

# data variables
word_to_index = vectorizer.vocabulary_
index_to_word = {i:t 
for t,i in word_to_index.items()}
tea_to_index = {tea: index for index, tea in enumerate(d["tea_category"] for d in tea_data)}

# get tea tfidf 
def get_tea_tfidf(teas):
    tfidf_vec = np.zeros(docs_compressed_normed.shape[1])
    for tea in teas: 
        tfidf_vec += docs_compressed_normed[tea_to_index[tea], :]
    return tfidf_vec / len(teas)

# get description tfidf 
def get_description_tfidf(description):
    description_tfidf = vectorizer.transform([description]).toarray()
    description_vec = normalize(np.dot(description_tfidf, words_compressed)).squeeze()
    return description_vec

# get query tfidf
def get_query_tfidf(search_teas, search_description):
    tfidf_vec = np.zeros(docs_compressed_normed.shape[1])

    entered_searches = 0
    if search_teas:
        tfidf_vec += get_tea_tfidf(search_teas)
        entered_searches += 1
    if search_description:
        tfidf_vec += get_description_tfidf(search_description)
        entered_searches += 1
    
    if entered_searches < 1: 
        return tfidf_vec

    return tfidf_vec / entered_searches

# get recommendations
def get_k_recommendations(search_teas, search_description, k=10, caffeine_options=["low", "moderate", "high"]):
    query_tfidf = get_query_tfidf(search_teas, search_description)
    sims = docs_compressed_normed.dot(query_tfidf)
    ranked_ids = (-sims).argsort()

    if search_teas: 
        search_tea_ids = [tea_to_index[t] for t in search_teas]
        ranked_ids = ranked_ids[~np.in1d(ranked_ids, search_tea_ids)] # remove the current searches
        
    data = []

    result_idx = 0
    results_added = 0

    while results_added < k:
        tea_id = ranked_ids[result_idx]
        if (tea_data[tea_id]["caffeine"] in caffeine_options):
            data.append({
                "tea_category": tea_data[tea_id]["tea_category"],
                "tea_type": tea_data[tea_id]["tea_type"],
                "about": tea_data[tea_id]["about"],
                "brands": tea_data[tea_id]["top_rated_brands"],
                "caffeine": tea_data[tea_id]["caffeine"],
                "score": sims[tea_id] 
            })
            results_added += 1
        result_idx += 1

    result = { "data": data }

    return json.dumps(result)
