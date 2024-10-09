# **Twitter Sentiment Analysis TF-IDF**

## **Introduction**

This project implements a Twitter Sentiment Analysis model utilizing the TF-IDF (Term Frequency-Inverse Document Frequency) technique, integrated with a FastAPI framework. It predicts the sentiment of a tweet as either positive, negative, or neutral.

## **How to Use**

To install the Twitter Sentiment Analysis TF-IDF follow these steps:

1. Clone the repository.
2. Install the required dependencies `pip install -r requirements.txt`
3. Run the FastAPI server using this command `python .\api.py`.
4. Use the /predict-sentiment endpoint to get sentiment predictions.
5. Open your browser and navigate to: **`http://127.0.0.1:8000/docs`**

## **API Endpoints**

## POST `/predict`

**Request Body**:

```json
POST /predict
{
  "message": "Your performance has been unsatisfactory. You missed 3 deadlines last month, and your error rate increased by 25%. This is unacceptable! 😡 #Disappointed"
}

```

**Response**

```
{
  "Response": "Negative"
}

```

## **License**

The Twitter Sentiment Analysis TF-IDF is released under the MIT License. See the **[LICENSE](https://github.com/DaramNikhil/sentiment-analysis-TF-IDF-Fastapi.git/blob/main/LICENSE)** file for details.

## **Hire Me**

If you're interested in hiring me for a software development, data science, or machine learning role, feel free to reach out!

-   **Author**: Daram Nikhil
-   **Email**: [nikhildaram51@gmail.com](mailto:nikhildaram51@gmail.com)
-   **GitHub**: [DaramNikhil](https://github.com/DaramNikhil)
-   **LinkedIn**: [daramnikhil](https://www.linkedin.com/in/daramnikhil)
