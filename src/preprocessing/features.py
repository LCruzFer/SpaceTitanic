from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer

def impute(data, method: str = 'median', **impute_kawrgs):
    '''
    Impute missing values using specified methods.
    '''
    imputer = SimpleImputer(method=method)
    imputer.fit(data)
    
    return imputer.transform(data)

def scaling(data, method: str = "standardize"):
    '''
    Scale provided data using specified method.
    '''
    if method == "standardize":
        scaler = StandardScaler()
    elif method == "normalize":
        scaler = Normalizer()
    else:
        raise TypeError(f'{method} does not exist. Must be one of standardize or normalize.')
    #fit to data and return transformed data
    scaler.fit(data)
    
    return scaler.transform(data)


#set filepath and names
data_in = Path.cwd()/'data'/'raw'
train_fname = "train.csv"
data_out = Path.cwd()/'data'/'processed'

#read in data
train = pd.read_csv(data_in/train_fname)
