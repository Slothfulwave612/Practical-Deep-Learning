"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module for data-augmentation.
"""

# required packages/modules
import numpy as np
import pandas as pd
from PIL import Image

import torchvision

# default value
RANDOM_SEED = 42


def augment_data(df, target):
    """
    Function to augment the data and add it to the
    train data set.

    Args:
        df (pandas.DataFrame): train data set.
        target (str): target column name.

    Returns:
        pandas.DataFrame: train set with augmented data.
    """
    # create an empty dataframe
    data_aug = pd.DataFrame()

    for label in sorted(df[target].unique()):

        # randomly select images from train dataframe
        images = df.loc[
            df[target] == label
        ].sample(n=200, random_state=RANDOM_SEED).drop(target, axis=1).values

        # create transforms
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.GaussianBlur(kernel_size=1),
            torchvision.transforms.RandomRotation(degrees=25),
            torchvision.transforms.RandomAffine(degrees=21, scale=(1, 1.2))
        ])

        # create an empty dataframe
        temp_df = pd.DataFrame()

        # traverse the images and make the required changes
        for image in images:
            # reshape the image and convert pixel values to original form
            image = image.reshape(28, 28) * 255

            # change the datatype
            image = image.astype(np.uint8)

            # convert to PIL image
            image = Image.fromarray(image, 'L')

            # apply data trnasformation
            image = data_transforms(image)

            # convert back to numpy
            image = np.array(image).reshape(28*28) / 255

            # conver to a dataframe
            image = pd.DataFrame(image).T

            # concat the dataframes
            temp_df = pd.concat(
                [temp_df, image], axis=0
            )

        # set label
        temp_df["label"] = label

        # concat the dataframes
        data_aug = pd.concat([
            data_aug, temp_df
        ], axis=0)

    # sample the dataframe
    data_aug = data_aug.sample(
        frac=1, random_state=RANDOM_SEED
    )

    # init an empty dict
    rename_dict = dict()

    # make the required dict
    for name in [f'pixel{i}' for i in data_aug.columns[:-1]]:
        rename_dict[int(name[5:])] = name

    # rename the columns
    data_aug.rename(rename_dict, inplace=True, axis=1)

    # concat to train data set
    df = pd.concat([
        df, data_aug
    ], axis=0).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # load the train data set
    train_df = pd.read_pickle("data/train.pkl")

    # augment the data
    train_df = augment_data(train_df, "label")

    # shape of the new dataframe
    print(train_df.shape)

    # save the dataframe
    train_df.to_pickle("data/train_aug.pkl")
