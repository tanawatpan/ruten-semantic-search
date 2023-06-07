import findspark
findspark.init()

from flask import Flask, request, jsonify
from flask_cors import CORS

from sentence_transformers import SentenceTransformer, util
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

import pyarrow.parquet as pq
import pandas as pd
import numpy as np


app = Flask(__name__)
CORS(app)

spark = SparkSession.builder \
                    .appName('Robert Sentence Embedding') \
                    .config("spark.dynamicAllocation.enabled", True) \
                    .config("spark.driver.memory", "4g") \
                    .config("spark.driver.maxResultSize", "4g") \
                    .config("spark.cores.max", 4) \
                    .config("spark.executor.cores", 1) \
                    .config("spark.executor.memory", "4g") \
                    .enableHiveSupport() \
                    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

MODEL = "mpnet"
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

items_df = spark.read.table(f"ruten.item_{MODEL}_embeddings").toPandas()
ids = items_df.item_id.values
sellers = items_df.seller_nickname.values
categories = items_df.category_name.values
item_names = items_df.item_name.values
embeddings = np.vstack(items_df.embedding.values)


def item_semantic_search(query, n=5):
    query_embeddings = model.encode(query)
    similarity = util.cos_sim(query_embeddings, embeddings)[0]
    top_results_indices = np.argsort(similarity)[-n:]
    return pd.DataFrame(
        zip(
            categories[top_results_indices],
            ids[top_results_indices],
            item_names[top_results_indices],
            np.array(similarity[top_results_indices])
        ),
        columns=["category", "item_id", "item_name", "similarity"]
    )


category_df = spark.read.table(f"ruten.category_{MODEL}_mean").toPandas()
category_names = category_df.category_name.values
category_means = np.vstack(category_df.means.values).astype(np.float32)


def category_semantic_search(query, n=5):
    query_embeddings = model.encode(query)
    similarity = util.cos_sim(query_embeddings, category_means)[0]
    top_results_indices = np.argsort(similarity)[-n:]
    return pd.DataFrame(
        zip(
            category_names[top_results_indices],
            np.array(similarity[top_results_indices])
        ),
        columns=["category_name", "similarity"]
    )


seller_df = spark.read.table(f"ruten.seller_{MODEL}_mean").toPandas()
seller_names = seller_df.seller_nickname.values
seller_means = np.vstack(seller_df.means.values).astype(np.float32)


def seller_semantic_search(query, n=5):
    query_embeddings = model.encode(query)
    similarity = util.cos_sim(query_embeddings, seller_means)[0]
    top_results_indices = np.argsort(similarity)[-n:]
    return pd.DataFrame(
        zip(
            seller_names[top_results_indices],
            np.array(similarity[top_results_indices])
        ),
        columns=["seller_name", "similarity"]
    )


@app.route('/item/search', methods=['GET'])
def item_search():
    n = request.args.get('n', 5, type=int)
    query = request.args.get('query', None)
    if query is None:
        return jsonify({"message": "Missing query in the request"}), 400
    return item_semantic_search(query, n).to_json(orient="records")


@app.route('/category/search', methods=['GET'])
def category_search():
    n = request.args.get('n', 5, type=int)
    query = request.args.get('query', None)
    if query is None:
        return jsonify({"message": "Missing query in the request"}), 400
    return category_semantic_search(query, n).to_json(orient="records")


@app.route('/seller/search', methods=['GET'])
def seller_search():
    n = request.args.get('n', 5, type=int)
    query = request.args.get('query', None)
    if query is None:
        return jsonify({"message": "Missing query in the request"}), 400
    return seller_semantic_search(query, n).to_json(orient="records")


if __name__ == '__main__':
    app.run(debug=False)
