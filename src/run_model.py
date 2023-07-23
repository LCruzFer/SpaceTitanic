from pathlib import Path
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from model.training_functions import evaluate
import datetime

# set paths
data_in = Path.cwd().parent / "data" / "processed"
data_out = Path.cwd().parent / "data" / "predictions"
model_path = Path.cwd().parent / "models"

# read in data
train_df = pd.read_csv(data_in / "train_preprocessed.csv", sep=";")
#!cleaning steps must be moved to preprocessing steps later
train_df = train_df.drop("Unnamed: 0", axis=1)
cols = train_df.columns
train_df = train_df.rename(columns={i: i.split("__")[1] for i in cols})
# replace whitespace with underline
train_df.columns = train_df.columns.str.replace(" ", "_")
train_df["CryoSleep"] = train_df["CryoSleep"].astype("bool")
train_df["VIP"] = train_df["VIP"].astype("bool")
train_df = train_df.drop(["PassengerId"], axis=1)

# test data
test_df = pd.read_csv(data_in / "test_preprocessed.csv", sep=";")
test_df = test_df.drop("Unnamed: 0", axis=1)
cols = test_df.columns
test_df = test_df.rename(columns={i: i.split("__")[1] for i in cols})
# replace whitespace with underline
test_df.columns = test_df.columns.str.replace(" ", "_")
test_df["CryoSleep"] = test_df["CryoSleep"].astype("bool")
test_df["VIP"] = test_df["VIP"].astype("bool")
test_df = test_df.drop(["PassengerId"], axis=1)

# define list of features
num_vars = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
cat_vars = [
    var for var in train_df.columns if ("Destination" in var) or ("HomePlanet" in var)
]
bool_vars = ["CryoSleep", "VIP"]
features = num_vars + cat_vars + bool_vars
# define target
target = ["Transported"]


# create validation set
test_df, valid_df = train_test_split(
    test_df, shuffle=False, random_state=2023, test_size=0.8
)
# split upt into X and y
X_train = train_df[features]
y_train = train_df[target]
# create X and y for validation and test set
X_valid = valid_df[features]
y_valid = valid_df[target]
X_test = test_df[features]
y_test = test_df[target]

# create LGB Datasets
train_dataset = lgb.Dataset(X_train, label=y_train)
valid_dataset = lgb.Dataset(X_valid, label=y_valid)
test_dataset = lgb.Dataset(X_test, label=y_test)

# set up model parameters
model_params = {"objective": "binary", "metric": "binary_logloss"}
num_round = 10000

# train model and predict values of validation set
bst = lgb.train(model_params, train_dataset, num_round, valid_sets=[test_dataset])
preds = bst.predict(X_test, prediction_type="class")
# convert probabilities to binary class
preds_binary = (preds > 0.5).astype(int)

# evaluate model
eval_scores = evaluate(test_df[target], preds_binary)
eval_scores_df = pd.DataFrame.from_dict(eval_scores, orient="index", columns=["score"])
print(eval_scores)