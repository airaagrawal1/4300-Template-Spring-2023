from collections import defaultdict
from collections import Counter
import json
import os
import math
import string
import time
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from IPython.core.display import HTML
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = ""
MYSQL_PORT = 3306
MYSQL_DATABASE = "teadb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# load the data
with open("tea_data.json", "r") as f: 
    tea_data = json.load(f)["data"]

# tokenize the data
tokenizer = TreebankWordTokenizer()
for tea in tea_data:
    tea['about_toks'] = tokenizer.tokenize(tea['about'])
    reviews_acc = "".join(tea['reviews'])
    tea['review_toks'] = tokenizer.tokenize(reviews_acc)

# constants
tea_categories = [tea["tea_category"] for tea in tea_data]
num_teas = len(tea_data)

adj_chars = [('a', 'q'), ('a', 's'), ('a', 'z'), ('b', 'g'), ('b', 'm'), ('b', 'n'), ('b', 'v'), ('c', 'd'),
             ('c', 'v'), ('c', 'x'), ('d', 'c'), ('d', 'e'), ('d', 'f'), ('d', 's'), ('e', 'd'), ('e', 'r'),
             ('e', 'w'), ('f', 'd'), ('f', 'g'), ('f', 'r'), ('f', 'v'), ('g', 'b'), ('g', 'f'), ('g', 'h'),
             ('g', 't'), ('h', 'g'), ('h', 'j'), ('h', 'm'), ('h', 'n'), ('h', 'y'), ('i', 'k'), ('i', 'o'),
             ('i', 'u'), ('j', 'h'), ('j', 'k'), ('j', 'u'), ('k', 'i'), ('k', 'j'), ('k', 'l'), ('l', 'k'),
             ('l', 'o'), ('m', 'b'), ('m', 'h'), ('n', 'b'), ('n', 'h'), ('o', 'i'), ('o', 'l'), ('o', 'p'),
             ('p', 'o'), ('q', 'a'), ('q', 'w'), ('r', 'e'), ('r', 'f'), ('r', 't'), ('s', 'a'), ('s', 'd'),
             ('s', 'w'), ('s', 'x'), ('t', 'g'), ('t', 'r'), ('t', 'y'), ('u', 'i'), ('u', 'j'), ('u', 'y'), 
             ('v', 'b'), ('v', 'c'), ('v', 'f'), ('w', 'e'), ('w', 'q'), ('w', 's'), ('x', 'c'), ('x', 's'), 
             ('x', 'z'), ('y', 'h'), ('y', 't'), ('y', 'u'), ('z', 'a'), ('z', 'x')]

# edit distance
def insertion_cost(text, j): 
    return 1

def deletion_cost(query, j):
    return 1

def substitution_cost(query, text, i, j):
    if query[i-1] == text[j-1]:
        return 0
    else:
        return 1

def substitution_cost_adj(query, text, i, j):
    a, b = query[i - 1], text[j - 1]
    if (a == b): 
        return 0
    elif ((a, b) in adj_chars):
        return 1.5
    else: 
        return 2

def edit_matrix(query, text, ins_cost_func, del_cost_func, sub_cost_func):
    m = len(query) + 1
    n = len(text) + 1

    chart = {(0, 0): 0}
    for i in range(1, m): 
        chart[i,0] = chart[i-1, 0] + del_cost_func(query, i) 
    for j in range(1, n): 
        chart[0,j] = chart[0, j-1] + ins_cost_func(text, j)
    for i in range(1, m):
        for j in range(1, n):
            chart[i, j] = min(
                chart[i-1, j] + del_cost_func(query, i),
                chart[i, j-1] + ins_cost_func(text, j),
                chart[i-1, j-1] + sub_cost_func(query, text, i, j)
            )
    return chart

def edit_distance(query, text, ins_cost_func, del_cost_func, sub_cost_func):
    query = query.lower()
    message = message.lower()
    
    m = len(query)
    n = len(message)
    edit_mat = edit_matrix(query, message, ins_cost_func, del_cost_func, sub_cost_func)
    return edit_mat[m, n]

def edit_distance_search(query, teas, ins_cost_func, del_cost_func, sub_cost_func):
    search_res = []
    tea_categories = [tea["tea_category"] for tea in teas]
    
    for category in tea_categories:
        edit_dist = edit_distance(query, category, ins_cost_func, del_cost_func, sub_cost_func)
        search_res.append((edit_dist, category))
    
    search_res.sort(key=lambda x: x[0])
    return search_res

# inverted index
def build_inverted_index(teas):
    inv_index = {}
    
    for doc_index in range(len(teas)): 
        tea = teas[doc_index]
        doc_dict = {} 
        tokens = tea['about_toks'] + tea['review_toks']
        
        for token in tokens:
            if token in doc_dict: 
                doc_dict[token] += 1
            else: 
                doc_dict[token] = 1
        
        for token, token_count in doc_dict.items(): 
            if token in inv_index: 
                inv_index[token].append((doc_index, token_count))
            else: 
                inv_index[token] = [(doc_index, token_count)]
            
    return inv_index

# compute idf
def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """
    returns dict such that for each term, the dict contains the idf value.
    """
    idf_idx = {}
    for term in inv_idx:
        df = len(inv_idx[term])
        if df >= min_df and (df / n_docs) <= max_df_ratio:
            idf_idx[term] = np.log2(n_docs / (1 + df))
    return idf_idx

# compute doc norms
def compute_doc_norms(index, idf, n_docs=num_teas):
    """
    index must be a dict, idf must be a dict.
    """
    result = np.zeros(n_docs)
    for word in idf:
        for doc, occur in index[word]:
            result[doc] += (occur * idf[word])**2 
    return np.sqrt(result)

# accumulate dot scores
def accumulate_dot_scores(query_word_counts, index, idf):
    """ computer numerator term for cosin similarity
    query_word_counts must be a dict (in the demo it will only be one word)"""
    result = {}
    for word in query_word_counts: 
        if word in index:
            for doc,count in index[word]: 
                if doc in result: 
                    result[doc]+= query_word_counts[word] * idf[word] * idf[word] * count 
                else: 
                    result[doc] = query_word_counts[word] * idf[word] * idf[word] * count 
    return result

# index search
def index_search(query, index, idf, doc_norms, score_func=accumulate_dot_scores, tokenizer=tokenizer):
    """returns a list of tuples (score, doc_id), a sorted list of results such that the first element has the 
    highest score, and `doc_id` points to the document with the highest score."""
    result = []
    tokens = tokenizer.tokenize(query.lower())
    query_dict = {}
    
    for word in tokens: 
        if word in index: 
            if word in query_dict: 
                query_dict[word]+=1
            else: 
                query_dict[word] = 1
    
    query_norm = 0 
    for word in tokens: 
        if word in idf: 
            query_norm += (query_dict[word] * idf[word])**2
    query_norm = math.sqrt(query_norm)
    
    num = score_func(query_dict, index, idf)
    for i in range(len(doc_norms)):
        if i in num:
            result.append(((num[i])/(query_norm * doc_norms[i]),i))
    return sorted(result, key = lambda x:x[0], reverse = True)

def sql_search(tea):
    # query_sql = f"""SELECT * FROM mytable WHERE LOWER( tea_category ) LIKE '%%{tea.lower()}%%' limit 10"""
    # keys = ["tea_category", "tea_type", "about"]
    # data = mysql_engine.query_selector(query_sql)
    inv_idx = build_inverted_index(tea_data)
    idf = compute_idf(inv_idx, num_teas, min_df=10, max_df_ratio=0.1)
    doc_norms = compute_doc_norms(inv_idx, idf, num_teas)

    data = []
    for _, tea_id in index_search(tea, inv_idx, idf, doc_norms)[:5]:
        data.append(tea_data[tea_id])
    return json.dumps(data)

@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text)

# app.run(debug=True)
