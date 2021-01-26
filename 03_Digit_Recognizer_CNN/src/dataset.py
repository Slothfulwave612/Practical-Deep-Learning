"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module for operating with dataset.
"""

import pandas as pd

# default value
RANDOM_SEED = 42

if __name__ == "__main__":
    # load the datasets
    train_data = pd.read_csv("input/train.csv")

    # shuffle the train dataset
    train_data = train_data.sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # normalize the data
    temp_data = train_data.loc[:, train_data.columns != "label"] / 255.0
    temp_data["label"] = train_data["label"]

    # create train, validation and test data set
    train, valid, test = temp_data.loc[0:39000].copy().reset_index(drop=True),\
        temp_data.loc[39001:40000].copy().reset_index(drop=True),\
        temp_data.loc[40001:].copy().reset_index(drop=True)

    # shape of datasets
    print(f"Shape of train data: {train.shape}")
    print(f"Shape of valid data: {valid.shape}")
    print(f"Shape of test data: {test.shape}")

    # save the datasets
    train.to_pickle("data/train.pkl")
    valid.to_pickle("data/valid.pkl")
    test.to_pickle("data/test.pkl")
