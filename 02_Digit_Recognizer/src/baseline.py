"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module for baseline model.
"""

import pandas as pd

from . import utils

if __name__ == "__main__":
    # load the dataset
    train_df = pd.read_pickle("data/modelling_data/train.pkl")
    valid_df = pd.read_pickle("data/modelling_data/valid.pkl")
    test_df = pd.read_pickle("data/modelling_data/test.pkl")

    # create mean values for pixel
    mean_tensor = utils.mean_values(train_df, "label")

    # create image from mean tensor
    fig, ax = utils.plot_mean_images(mean_tensor)

    # calculate accuracy on validation data
    train_accuracy = utils.calculate_accuracy_baseline(
        train_df, mean_tensor, "label"
    )
    valid_accuracy = utils.calculate_accuracy_baseline(
        valid_df, mean_tensor, "label"
    )
    test_accuracy = utils.calculate_accuracy_baseline(
        test_df, mean_tensor, "label"
    )

    print()
    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {valid_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
