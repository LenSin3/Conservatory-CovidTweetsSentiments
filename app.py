# Flask App
# Import dependencies
from flask import Flask, render_template, jsonify, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from config import DATABASE_URI

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import datetime

# import and load saved transformer and model

vector_path = "main_cnt_vec.pkl"
trans_path = "main_text3_tfidf.pkl"
model_path = "linearSVC.pkl"

with open(vector_path, "rb") as f:
    text_vector = pickle.load(f)

with open(trans_path, "rb") as f:
    text_transformer = pickle.load(f)

with open(model_path, "rb") as f:
    lsvc_model = pickle.load(f)



app = Flask(__name__)

#################################################
# Database Setup
#################################################

engine = create_engine(DATABASE_URI)
conn = engine.connect()


# create route that renders index.html template with prediction app
@app.route("/", methods=['GET', 'POST'])
def index():
    """Returns the homepage with Prediction App"""
    if request.method == 'GET':


        return render_template('index.html')
    
    if request.method == 'POST':

        message = request.form['text']
        data_vector = text_vector.transform([message])
        data_transform = text_transformer.transform(data_vector)
        prediction = lsvc_model.predict(data_transform)
        # output_prediction = lsvc_model.predict(data_transform)
        
        # return render_template('result.html', output_prediction = prediction)
        return render_template('index.html', output_prediction = prediction)

# create route to return unique dates
@app.route("/uniqueDates")
def uniqueDates():
    

    """Returns a list of unique dates"""
    df_dates = pd.read_sql_query('''SELECT DISTINCT(tweeet_date) from tweets_sentiment;''', conn)
    df_dates['dates'] = pd.to_datetime(df_dates['tweeet_date'], errors='coerce')
    df_dates['month'] = df_dates['dates'].dt.strftime('%b')
    df_dates_unq = df_dates['month'].unique()
    
    # Convert list of tuples into normal list
    # unique_dates = list(np.ravel(udates))
    return jsonify(list(df_dates_unq))

# create route to return sentiments
@app.route("/uniqueDates/<label>")
def getLabels(label):

    """Returns sentiment counts by date"""
    df_sent = pd.read_sql_query("""SELECT tweeet_date, sentiment FROM tweets_sentiment;""", conn)
    dfMain = df_sent.loc[df_sent['sentiment'] == label.title()]
    dfMain['dates'] = pd.to_datetime(dfMain['tweeet_date'], errors='coerce')
    dfMain['month'] = dfMain['dates'].dt.strftime('%b')
    dfLabel = dfMain.groupby('month').count().reset_index()
    
    
    

    label_count = []

    for index, row in dfLabel.iterrows():
        dfLabels = {}
        dfLabels['month'] = row['month']
        dfLabels['sentimentcount'] = row['sentiment']
        label_count.append(dfLabels)
        
    return jsonify(label_count)

# create route to filter by label and month
@app.route('/uniqueDates/<monthlab>')
def getmonthlabels(monthlab):

    """Returns Count of Labels in that month"""
    df_sent = pd.read_sql_query("""SELECT tweeet_date, tweet, sentiment FROM tweets_sentiment;""", conn)
    
    df_sent['dates'] = pd.to_datetime(df_sent['tweeet_date'], errors='coerce')
    df_sent['month'] = df_sent['dates'].dt.strftime('%b')
    dfMain = df_sent.loc[df_sent['month'] == monthlab.title()]
    dfLabel = dfMain.groupby('sentiment')['tweet'].count().reset_index()

    label_count = []

    for index, row in dfLabel.iterrows():
        dfLabels = {}
        dfLabels['sentiment'] = row['sentiment']
        dfLabels['tweetCount'] = row['tweet']
        label_count.append(dfLabels)
        
    return jsonify(label_count)


if __name__ == "__main__":

    app.run(debug=True)
