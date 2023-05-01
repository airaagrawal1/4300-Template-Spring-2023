import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from ir.edit_distance import top_k_edit_distance

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
def get_tea_tfidf(tea):
    return docs_compressed_normed[tea_to_index[tea], :]

# get description tfidf 
def get_description_tfidf(description):
    description_tfidf = vectorizer.transform([description]).toarray()
    description_vec = normalize(np.dot(description_tfidf, words_compressed)).squeeze()
    return description_vec

# get query tfidf
def get_query_tfidf(search_tea, search_description):
    tfidf_vec = np.zeros(docs_compressed_normed.shape[1])
    entered_searches = 0
    if search_tea:
        tfidf_vec += get_tea_tfidf(search_tea)
        entered_searches += 1
    if search_description:
        tfidf_vec += get_description_tfidf(search_description)
        entered_searches += 1
    
    if entered_searches < 1: 
        return tfidf_vec

    return tfidf_vec / entered_searches

# get recommendations
def get_k_recommendations(search_tea, search_description, k=10, cafArray=["low", "moderate", "high"]):
    query_tfidf = get_query_tfidf(search_tea, search_description)
    sims = docs_compressed_normed.dot(query_tfidf)
    ranked_ids = (-sims).argsort()

    if search_tea: 
        ranked_ids = ranked_ids[ranked_ids != tea_to_index[search_tea]] # remove the current search
        
    data = []
    for tea_id in ranked_ids[:k]:
        if (tea_data[tea_id]["caffeine"] in cafArray):
            data.append({
                "tea_category": tea_data[tea_id]["tea_category"],
                "tea_type": tea_data[tea_id]["tea_type"],
                "about": tea_data[tea_id]["about"],
                "brands": tea_data[tea_id]["top_rated_brands"],
                "caffeine": tea_data[tea_id]["caffeine"],
                "score": sims[tea_id] 
            })
        result = { "data": data }

    return json.dumps(result)
