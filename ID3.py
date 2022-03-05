import pandas as pd

data = pd.read_csv("dataset/data.csv")
print(data.head())
print(data['Edad'].unique())