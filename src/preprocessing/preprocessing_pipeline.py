from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

def rm_dups(df: pd.DataFrame):
    return df.drop_duplicates()

def rm_missing(df: pd.DataFrame):
    return df.dropna()

train = pd.read_csv('/Users/lucascruzfernandez/Documents/space_titanic/data/raw/train.csv')

#define lists of variables by type 
categoricals = ['HomePlanet', 'Cabin', 'Destination', 'Name', 'FirstName', 'LastName']
booleans = ['CryoSleep', 'VIP']
numericals = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
#define target variable
target = ['Transported']

#set keywords for imputation and scaling steps
impute_kwargs = {'missing_values': np.nan, 'strategy': 'median'}
scaler_kwargs = {}

ft = FunctionTransformer(rm_dups)
ct = ColumnTransformer(transformers=['impute', SimpleImputer(**impute_kwargs), ])

pipe = [('remove_duplicates', ct), 
        ('impute', SimpleImputer(**impute_kwargs)), 
        ('scaler', StandardScaler())]

pipeline = Pipeline(pipe)
