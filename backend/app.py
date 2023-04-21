import json
import os
import math
import string
import time
import re
import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from ir.recommendation import get_recommendations

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

# mysql_engine = MySQLDatabaseHandler(
#     MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE)

# # Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

def success_response(data, code=200):
    return json.dumps(data), code

def failure_response(message, code=404):
    return json.dumps({"error": message}), code

# load the data
with open("tea_data.json", "r") as f:
    tea_data = json.load(f)["data"]

# constants
tea_categories = [tea["tea_category"] for tea in tea_data]
num_teas = len(tea_data)


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/api/tea_names/")
def get_tea_names():
    return tea_categories

@app.route("/api/teas", methods=["POST"])
def get_teas():
    search_tea = request.args.get("tea")
    search_description = request.args.get("description")
    return get_recommendations(search_tea, search_description, 5)

# @app.route("/app/reviews/<tea_id:int>/")
def create_review():
    # TODO create a review for a tea
    pass


def create_relevant():
    # TODO allow users to mark relevant teas and then re-generate query results based on rocchio
    pass

# app.run(debug=True)
