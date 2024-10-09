from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from utils import load_models
from textblob import TextBlob
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, Welcome to the sentiment analysis API"}




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
    cl_new_data = clean_text(data)
    st_new_data = remove_stopwords(cl_new_data)
    return st_new_data

def prediction_data(data):
    tfidf = load_models(model_path=r"D:\my_projects\twitter-sentiment-analysis\models\tfidf_vectorizer.pkl")
    rf_model = load_models(model_path=r"D:\my_projects\twitter-sentiment-analysis\models\rf_model_sentiment_analysis.pkl")
    new_data_tfidf = tfidf.transform([data]).toarray()
    pred_emotion = rf_model.predict(new_data_tfidf)
    return pred_emotion


@app.post("/predict")
def predict(message):
    try:
        new_data = str(message)
        cleaned_data = data_preprocessing(new_data)
        pred_emotion = prediction_data(cleaned_data)

        if pred_emotion[0] == 3:
            return {"response": "Positive"}
        elif pred_emotion[0] == 2:
            return {"response": "Negative"}
        elif pred_emotion[0] == 1:
            return {"response": "Neutral"}
        else:
            return {"response": "Irrelevant"}

    except Exception as e:
        return {"error": str(e)}
    



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)