# Conservatory-Sentiment Analysis on COVID-19 Tweets
Develop Machine Learning models to decipher covid tweets and predict whether they are positive or negative tweets or neutral

People use social media not only to share information, but to share their feelings. Over the last year, Coronavirus and the resulting quarantine has greatly affected our lives, and social media
platforms, primarily Twitter, which are overflowing with posts about this topic. 
While positivity and negativity are blatantly obvious in some tweets, other times we can struggle to decipher sentiments in a loaded tweet. 

We used machine learning tools to execute sentiment analysis on COVID-19 related tweets starting with data extraction from Kaggle, uploaded into Jupyter notebook for clean up and sorting (filling NaN values with “Unknown” for those that were missing location data; drop unnecessary columns such as user names and screen names, checked that we had unique labels to ensure our datasets were clean.

We then selected Scikit-learn as our primary machine learning library because of its simple and effective nature for which we started with text processing, to  tokenizing the text, to then transform the data using TFIDF Transformer.
Our evaluation of the machine learning model started with the train-test split, and then used the following classification models to do predictions:
- Logistic regression 
- Linear SVC 
- Naive Bayes 
- Random Forest Classifier

![Snip20210210_2](https://user-images.githubusercontent.com/66816965/107607030-d2eaf700-6bec-11eb-9a92-43b9b7690ae4.png)
 
 We selected our second model LinearSVC as it yielded the highest accuracy.  We then saved CountVectorizer and TFID Transformer objects into a pickle file.  Also, saved a LinearSVC model with best parameters from the GridSearchCV.
 
 ![Snip20210210_3](https://user-images.githubusercontent.com/66816965/107607079-0463c280-6bed-11eb-882b-7b0b63b9c4a3.png)
  
We used Flask to render the result of submitted tweets. This will enable us to make real time predictions. We also extracted more data to test our model. Due to the large size of datasets, we decided to use about 0.5% of the new data which still had about 360000 rows. Thisdata together with train and test data were loaded in a postgresql database.

<picture>
