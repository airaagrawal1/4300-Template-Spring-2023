import numpy as np
import json
from ir.recommendation import get_query_tfidf, tea_to_index, docs_compressed_normed, tea_data

def rocchio(search_teas, search_description, relevant, irrelevant, input_doc_matrix=docs_compressed_normed, a=1, b=0.95, c=0.4, clip=True, k=10):
    q0 = get_query_tfidf(search_teas, search_description)

    rel_docs = np.zeros(len(q0))
    irrel_docs = np.zeros(len(q0))
    
    if relevant:
        rel_docs_i = [tea_to_index[rel_t] for rel_t in relevant]
        rel_docs = np.mean(input_doc_matrix[rel_docs_i], axis = 0) 

    if irrelevant:
        irrel_docs_i = [tea_to_index[irrel_t] for irrel_t in irrelevant]
        irrel_docs = np.mean(input_doc_matrix[irrel_docs_i], axis = 0) 

    q1 = a * q0 + b * rel_docs - c * irrel_docs
    if clip: q1[q1 < 0] = 0
        
    sims = input_doc_matrix.dot(q1)
    ranked_ids = (-sims).argsort()

    if search_teas: 
        search_tea_ids = [tea_to_index[t] for t in search_teas]
        ranked_ids = ranked_ids[~np.in1d(ranked_ids, search_tea_ids)] # remove the current searches
        
    data = []
    for tea_id in ranked_ids[:k]:
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

