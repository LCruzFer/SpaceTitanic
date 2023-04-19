import pandas as pd

def split_col(df: pd.DataFrame, split_col: str, delim: str, new_cols: list[str]):
    '''
    Split a column containing a string into to new columns.
    
    *df: pd.DataFrame
    *split_col: str
        Column that should be split.
    *delim: str
        Delimiter to split on.
    *new_cols: list[str]
        Names of new columns added by split.
    '''
    df[new_cols] = df[split_col].str.split(delim, n=1, expand=True)
    
    return df

def get_family_identifier(df: pd.DataFrame):
    '''
    Construct identifier for families. Passengers are considered to be part of the same family if they are part of the same group and have the same last name.
    '''
    #creating necessary columns 
    #! this step should be happening somewhere else later
    df = split_col(df, 'PassengerId', '_', ['gggg', 'pp'])
    df = split_col(df, 'Name', ' ', ['FirstName', 'LastName'])
    #create family ID
    df['FamilyId'] = df.groupby(['gggg', 'LastName']).cumcount()+1
    
    return df

def gen_features(df):
    '''
    Wrapper function to generate new features.
    '''
    df = get_family_identifier(df)
    
    return df
