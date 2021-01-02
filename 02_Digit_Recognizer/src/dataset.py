"""
__author__: @slothfulwave612

Python module for operating with dataset.
"""

import pandas as pd

if __name__ == "__main__":
    ## load the datasets
    train_data = pd.read_csv("input/digit-recognizer/train.csv")

    ## shuffle the train dataset
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    ## normalize the data
    temp_data = train_data.loc[:, train_data.columns != "label"] / 255.0
    temp_data["label"] = train_data["label"]

    ## create train, validation and test data set
    train, valid, test = temp_data.loc[0:38000].copy(), temp_data.loc[38001:40000].copy(), temp_data.loc[40000:].copy()

    ## shape of datasets
    print(f"Shape of train data: {train.shape}")
    print(f"Shape of valid data: {valid.shape}")
    print(f"Shape of test data: {test.shape}")

    ## save the datasets
    train.to_csv("data/train_data/train.csv", index=False)
    valid.to_csv("data/train_data/valid.csv", index=False)
    test.to_csv("data/train_data/test.csv", index=False)