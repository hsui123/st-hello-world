import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import streamlit as st
import pickle

st.header("My sentiment Model")

df = pd.read_csv("C:/Users/41946/Desktop/machine learning/text/IMDB_movie_reviews_labeled.csv")

st.subheader("Tranining Data Sample")

st.dataframe(df.sample(5))

st.write(df.sentiment.value_counts())

pipeline = None
if st.button("Build my machine learning pipeline"):
    X = df.loc[:, ['review']]

    y = df.sentiment

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)

    X_train_docs = [doc for doc in X_train.review]

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english',max_features=1000)),
                        ('cls', LinearSVC())])

    pipeline.fit(X_train_docs, y_train)

    Pipeline(steps=[('vect',TfidfVectorizer(max_features=1000, ngram_range=(1, 2),stop_words='english')),
                    ('cls', LinearSVC())])

    training_accuracy  = cross_val_score(pipeline, X_train_docs, y_train, cv=5).mean()

    predicted = pipeline.predict([doc for doc in X_test.review])

    validation_accuracy = accuracy_score(y_test, predicted)

    st.subheader('Model Performence')

    st.write("training accuracy",training_accuracy)

    st.write("validation accuracy ", validation_accuracy)
    with open('pipeline.pkl','wb') as f:
        pickle.dump(pipeline,f)

st.subheader("Testing the model")

revire_text = st.text_area('Movie Review')

if st.button("Predict"):
    with open('pipeline.pkl','rb') as f:
        pipeline = pickle.load(f)
    sentiment = pipeline.predict([revire_text])
    st.write("Predict sentiment is:", sentiment[0])

st.subheader('Titanic Model Testing')
pclass = st.selectbox('pclass', options = ['1','2','3'])
age = st.text_input('age')
if st.button('predict survival'):
    st.write('model predicts')
