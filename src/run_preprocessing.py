from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocessing_pipeline as pp

# Path options and filenames
data_in = Path.cwd().parent / "data" / "raw"
data_out = Path.cwd().parent / "data" / "processed"

raw_data_fname = "train.csv"
train_out_fname = "train_preprocessed.csv"
test_out_fname = "test_preprocessed.csv"

if __name__ == "__main__":
    # * general options
    rand_state = 2023

    # define lists containing column names by type
    num_vars = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    #Ignoring "Name" & "Cabin" feature for now until having a better idea how to use it
    cat_vars = ["HomePlanet", "Destination"]
    bool_vars = ["CryoSleep", "VIP"]

    # * Pipeline Options 
    #should missings just be removed?
    clean_missings_option = False
    #kwargs for sklearn functions
    impute_kwargs = {"strategy": "median"}
    scaler_kwargs = {}
    enc_kwargs = {}

    # * read data
    raw_df = pd.read_csv(data_in / raw_data_fname)
    #! ignoring Name & Cabin feature for now
    raw_df = raw_df.drop(["Name", "Cabin"], axis=1)
    # split data into training and test set for model training
    # shuffle = False prevents train_test_split from splitting df into X and y already, since writing results to disk and reading in later, this will presumably save computing time
    raw_df_shuffled = raw_df.sample(frac=1, random_state=rand_state)
    train_raw, test_raw = train_test_split(
        raw_df_shuffled, shuffle=False, random_state=rand_state
    )

    # build complete sklearn pipeline instances
    clean_pipe = pp.setup_clean_pipeline(missings=clean_missings_option)
    num_pipe = pp.setup_num_pipeline(impute_kwargs, scaler_kwargs)
    cat_pipe = pp.setup_cat_pipeline(enc_kwargs)
    comp_pipe = pp.setup_complete_pipeline(
        clean_pipe, num_pipe, cat_pipe, num_vars, cat_vars
    )

    # pass data through pipeline
    train_pp = comp_pipe.fit_transform(train_raw)
    test_pp = comp_pipe.fit_transform(test_raw)

    #retrieve feature names and assign
    feat_names = comp_pipe[-1].get_feature_names_out()
    train_pp_df = pd.DataFrame(train_pp, columns=feat_names)
    test_pp_df = pd.DataFrame(test_pp, columns=feat_names)

    # save results
    train_pp_df.to_csv(data_out / train_out_fname, sep=";")
    test_pp_df.to_csv(data_out / test_out_fname, sep=";")