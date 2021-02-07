# Flask App
# Import dependencies
from flask import Flask, render_template, jsonify, redirect, url_for, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

# import and load saved transformer and model

trans_path = "Conservatory-Product-Review-Classififcation/corona_tweets/new_text3_tfidf.pkl"
model_path = "Conservatory-Product-Review-Classififcation/corona_tweets/new_grid3_lsvc.pkl"
with open(trans_path, "rb") as f:
    text_transformer = pickle.load(f)

with open(model_path, "rb") as f:
    lsvc_model = pickle.load(f)





app = Flask(__name__)

# create route that renders index.html template
@app.route("/", methods=['GET', 'POST'])
def index():
    """Returns the homepage"""
    if request.method == 'GET':


        return render_template('index.html')
    
    if request.method == 'POST':

        message = request.form['text']
        data = [message]
        data_transform = text_transformer.transform(data)
        prediction = lsvc_model.predict(data_transform)
    return render_template('index.html', output_prediction = prediction)
	

if __name__ == "__main__":

    app.run(debug=True)
