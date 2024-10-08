import pandas as pd
import numpy as np

data = pd.read_csv(r"D:\my_projects\twitter-sentiment-analysis\data\twitter_training.csv")
main_data = data[:1000]
main_data.to_csv(r"main_data.csv", index=False)
print("success")