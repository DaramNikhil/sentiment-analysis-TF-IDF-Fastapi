import pandas as pd
import numpy as np

def Data_Load():
    data = pd.read_csv(r"D:\my_projects\twitter-sentiment-analysis\data\twitter_training.csv")
    main_data = data[:1000]
    main_data.to_csv(r"D:\my_projects\twitter-sentiment-analysis\data\main_data\main_data.csv", index=False)

Data_Load()