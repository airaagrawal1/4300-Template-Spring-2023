# import libraries 
import json
import re

with open("./../tea_data.json", "r") as file: 
    data = json.load(file)["data"]

# print(data)

for tea in data:
    tea["doc"] = tea["about"] + " ".join(tea["reviews"]) 

tea_json = { "data": data }
json_data = json.dumps(tea_json)

with open("./../tea_data.json", "w") as file: 
    file.write(json_data)