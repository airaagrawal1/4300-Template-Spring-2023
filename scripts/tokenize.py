# import libraries 
import json
import re

# with open("./../tea_data.json", "r") as file: 
#     data = json.load(file)["data"]

# print(data)

# tokenizer 
def tokenize(text): 
    return [x for x in re.findall(r"[a-z]+", text.lower())]

# tokenize the data
def tokenize_data(data):
    for tea in data:
        tea['about_toks'] = tokenize(tea['about'])
        reviews_acc = "".join(tea['reviews'])
        tea['review_toks'] = tokenize(reviews_acc)

# tea_json = { "data": data }
# json_data = json.dumps(tea_json)

# with open("./../tea_data.json", "w") as file: 
#     file.write(json_data)