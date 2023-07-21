from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


def rm_duplicates(df: pd.DataFrame):
    return df.drop_duplicates()


def rm_missings(df: pd.DataFrame):
    return df.dropna()


def setup_clean_pipeline(missings: bool = False):
    """
    Set up pipeline to clean data before other processing steps.

    This is currently iteration 1: removing duplicates and decide if missings should just be removed
    """
    if missings:        
        clean_pipeline = Pipeline(
            [
                ("rm_dups", FunctionTransformer(rm_duplicates)),
                ("rm_missings", FunctionTransformer(rm_missings)),
            ]
        )
    else:
        clean_pipeline = Pipeline(
            [
                ("rm_dups", FunctionTransformer(rm_duplicates)),
            ]
        )

    return clean_pipeline


def setup_num_pipeline(impute_kwargs: dict = {}, scaler_kwargs: dict = {}):
    """
    Set up pipeline to process numerical data.
    Steps taken:
        - imputation of missing values
        - scaling

    *impute_kwargs: dict
        Arguments to pass to imputation function.
    *scaler_kwargs: dict
        Arguments to pass to scaling function.
    """
    numerical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(**impute_kwargs)),
            ("scale", StandardScaler(**scaler_kwargs)),
        ]
    )

    return numerical_pipeline


def setup_cat_pipeline(enc_kwargs):
    """
    Set up pipeline to process categorical data.
    Steps taken:
        - one hot encoding of categorical features

    *enc_kwargs: dict
        Arguments to pass to encoding function.
    """
    cat_enc = Pipeline([("encoder", OneHotEncoder(**enc_kwargs))])

    return cat_enc


def setup_complete_pipeline(clean_pipe, num_pipe, cat_pipe, num_vars, cat_vars):
    """
    Set up complete preprocessing pipeline object.

    *clean_pipe: sklearn.pipeline.Pipeline
        Pipeline  to clean data.
    *num_pipe: sklearn.pipeline.Pipeline
        Pipeline to handle numerical data.
    *cat_pipe: sklearn.pipeline.Pipeline
        Pipeline to handle categorical data
    """
    comp_pipeline = Pipeline(
        [
            #("cleaning", clean_pipe),
            (
                "column_specific",
                ColumnTransformer(
                    [
                        ("numerical", num_pipe, num_vars),
                        ("categorical", cat_pipe, cat_vars),
                    ],
                    remainder="passthrough",
                ),
            ),
        ]
    )

    return comp_pipeline
