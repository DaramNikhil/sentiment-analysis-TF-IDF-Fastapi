import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


def clean_text(text):
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+|[^A-Za-z\s]', '', text)
    text = text.lower()
    text = str(TextBlob(text).correct())
    return text

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)

def data_preprocessing(data):
    data.dropna(inplace=True)
    data.drop(["2401","Borderlands"],axis=1, inplace=True)
    data.rename({"Positive": "emotion", "im getting on borderlands and i will murder you all ,": "message"}, axis=1, inplace=True)
    data["message"] = data["message"].apply(clean_text)
    data["message"] = data["message"].apply(remove_stopwords)
    return data

def prediction_data(data):
    le = LabelEncoder()
    data["emotion"] = le.fit_transform(data["emotion"])
    tfidf_vc = TfidfVectorizer(max_features=1000)
    X = tfidf_vc.fit_transform(data["message"]).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

    



if __name__ == '__main__':
    data = r"D:\my_projects\twitter-sentiment-analysis\data\main_data\main_data.csv"
    data = pd.read_csv(data)
    cleaned_data = data_preprocessing(data)
    model_score = prediction_data(cleaned_data)
    print(f"Model Score: {model_score*100}")