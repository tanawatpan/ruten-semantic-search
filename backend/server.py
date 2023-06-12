# %pip install torch --index-url https://download.pytorch.org/whl/cpu
# %pip install sentence-transformers
# %pip install flask flask-cors

import findspark
findspark.init()

from flask import Flask, request, jsonify
from flask_cors import CORS

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from transformers import BertTokenizerFast, TFBertModel, BertConfig
from sentence_transformers import util

import pyarrow.parquet as pq
import tensorflow as tf
import pandas as pd
import numpy as np
import subprocess

app = Flask(__name__)
CORS(app)

spark = SparkSession.builder \
                    .appName('BERT Sentence Embedding') \
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

BASE = "hfl/chinese-macbert-base"
MODEL = "bert_model_base.h5"
subprocess.run(["hdfs", "dfs", "-copyToLocal", f"/ruten/{MODEL}", "./"], shell=True)
tokenizer = BertTokenizerFast.from_pretrained(BASE)
model = TFBertModel.from_pretrained(MODEL, config=BertConfig.from_pretrained(BASE))


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)
    sum_embeddings = tf.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = tf.math.reduce_sum(input_mask_expanded, axis=1)
    sentence_embeddings = sum_embeddings / tf.math.maximum(sum_mask, 1e-9)
    return tf.math.l2_normalize(sentence_embeddings, axis=1)


def encode(query):
    encoded_input = tokenizer(
        query,
        max_length=128, 
        padding=True,
        truncation=True, 
        return_tensors="tf"
    )
    model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).numpy()


def item_semantic_search(query, n=5):
    query_embeddings = encode(query)
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


def category_semantic_search(query, n=5):
    query_embeddings = encode(query)
    similarity = util.cos_sim(query_embeddings, category_means)[0]
    top_results_indices = np.argsort(similarity)[-n:]
    return pd.DataFrame(
        zip(
            category_names[top_results_indices],
            np.array(similarity[top_results_indices])
        ),
        columns=["category_name", "similarity"]
    )


def seller_semantic_search(query, n=5):
    query_embeddings = encode(query)
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
    items_df = spark.read.table(f"ruten.item_bert_embeddings").toPandas()
    ids = items_df.item_id.values
    sellers = items_df.seller_nickname.values
    categories = items_df.category_name.values
    item_names = items_df.item_name.values
    embeddings = np.vstack(items_df.embedding.values)

    category_df = spark.read.table(f"ruten.category_bert_mean").toPandas()
    category_names = category_df.category_name.values
    category_means = np.vstack(category_df.means.values).astype(np.float32)

    seller_df = spark.read.table(f"ruten.seller_bert_mean").toPandas()
    seller_names = seller_df.seller_nickname.values
    seller_means = np.vstack(seller_df.means.values).astype(np.float32)

    app.run(host='0.0.0.0', debug=False)
