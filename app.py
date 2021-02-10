# Flask App
# Import dependencies
from flask import Flask, render_template, jsonify, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC




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


if __name__ == "__main__":

    app.run(debug=True)
