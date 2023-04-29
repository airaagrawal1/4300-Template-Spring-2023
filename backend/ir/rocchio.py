import numpy as np
import json
from ir.recommendation import get_query_tfidf, tea_to_index, docs_compressed_normed, tea_data
from ir.edit_distance import top_k_edit_distance

def rocchio(search_tea, search_description, relevant, irrelevant, input_doc_matrix=docs_compressed_normed, a=0.3, b=0.3, c=0.8, clip=True):
    q0 = get_query_tfidf(search_tea, search_description)
    
    rel_docs_i = [tea_to_index[rel_t] for rel_t in relevant]
    rel_docs = np.mean(input_doc_matrix[rel_docs_i], axis = 0) if rel_docs_i != [] else 0

    irrel_docs_i = [tea_to_index[irrel_t] for irrel_t in irrelevant]
    irel_docs = np.mean(input_doc_matrix[irrel_docs_i], axis = 0) if irrel_docs_i != [] else 0

    q1 = a * q0 + b * rel_docs - c * irel_docs
    if clip: q1[q1 < 0] = 0
        
    sims = docs_compressed_normed.dot(q1)
    ranked_ids = (-sims).argsort()

    if search_tea: 
        ranked_ids = ranked_ids[ranked_ids != tea_to_index[search_tea]] # remove the current search
        
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

