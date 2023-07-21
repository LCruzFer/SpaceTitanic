import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv(
    "/Users/lucascruzfernandez/Documents/space_titanic/data/processed/train_preprocessed.csv",
    sep=";",
)
train = train.drop(["Unnamed: 0"], axis=1)

