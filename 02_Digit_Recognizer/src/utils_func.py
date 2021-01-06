"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module containing utility functions.
"""
# required packages/modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
from tqdm import tqdm

from sklearn import metrics

import torch
import torch.nn as nn


def mean_values(df, label):
    """
    Function to yield mean values for
    each pixel for each digit present
    in the dataset.

    Args:
        df (pandas.DataFrame): required dataframe.
        label (str): label column name.

    Returns:
        torch.tensor: mean value for each pixel.
    """
    # all unique labels
    labels = sorted(df[label].unique())

    # empty array for storing mean values
    mean_list = []

    for digit in labels:
        # calculate mean value for each pixel for every image
        mean_value = torch.tensor(
            df.loc[
                df[label] == digit
            ].drop(label, axis=1).values
        ).mean(0)

        mean_list.append(mean_value)

    # convert list to tensor
    mean_tensor = torch.stack(mean_list)

    return mean_tensor


def plot_mean_images(mean_tensor):
    """
    Function to produce images using values
    from mean_tensor.

    Args:
        mean_tensor (torch.tensor): mean value for each pixel.

    Returns:
        figure.Figure: figure object.
        axes.Axes: axes object.
    """
    # create a subplot
    fig, axes = plt.subplots(
        nrows=2, ncols=5, facecolor="#121212", figsize=(16, 8)
    )

    # color map
    cmap = [
        "#121212", "#F2F2F2"
    ]
    cmap = colors.ListedColormap(cmap)

    # traverse axes and plot mean_tensor image for every digit
    for index, ax in enumerate(fig.get_axes()):

        # set facecolor
        # ax.set_facecolor("#121212")

        # plot the image
        ax.imshow(
            mean_tensor[index].reshape(28, 28), cmap="Greys"
        )

        # tidy axis
        ax.axis("off")

    # add figure title
    fig.suptitle(
        "Baseline model images on train data", fontsize=20, color="#F2F2F2"
    )

    fig.savefig("plots/baseline_img_train.jpg", dpi=500, bbox_inches="tight")

    return fig, axes


def calculate_mae(pred, true):
    """
    Function to calculate MAE value.

    Args:
        pred (torch.tensor): predicted values.
        true (torch.tensor): true values.

    Returns:
        float: MAE score.
    """
    # initialise object of L1Loss
    mae_loss = nn.L1Loss()

    # set requires_grad to True
    pred.requires_grad_(True)

    # calculate the loss
    output = mae_loss(pred, true)

    return round(output.item(), 3)


def calculate_accuracy_baseline(df, mean_tensor, label):
    """
    Function to predict label using mean_tensor value.

    Args:
        df (pd.DataFrame): required dataframe.
        mean_tensor (torch.tensor): mean value for each pixel.
        label (str): label column name.

    Returns:
        float: accuracy on the data set passed.
    """
    # data required
    data = torch.from_numpy(
        df.drop(label, axis=1).values
    )

    # true labels
    labels = df[label].values

    # empty list for storing predicted labels
    pred_labels = []

    # traverse the data
    for image in tqdm(data, total=len(data)):

        # empty list which will contain MAE values
        mae_scores = []

        # traverse the mean value
        for means in mean_tensor:

            # calculate MAE value
            output = calculate_mae(means, image)

            # append to list
            mae_scores.append(output)

        # append predicted label
        pred_labels.append(
            np.array(mae_scores).argmin()
        )

    return round(
        metrics.accuracy_score(
            pred_labels, labels
        ), 3
    )


def plot_over_epochs(
    train_list, valid_list, title=None, ylabel=None, path=None
):
    """
    Function to plot a line plot for train and valid stats.

    Args:
        train_list (list): containing training stats.
        valid_list (list): containing validation stats.
        title (str, optional): title of the plot. Defaults to None.
        ylabel (str, optional): ylabel for the plot. Defaults to None.
        path (str, optional): path where plot will be saved. Defaults to None.

    Returns:
        figure.Figure: figure object.
        axes.Axes: axes object.
    """
    # default font-family
    rcParams["font.family"] = "serif"

    # create subplot
    fig, ax = plt.subplots(facecolor="#F2F2F2", figsize=(12, 8))
    ax.set_facecolor("#F2F2F2")

    # plot train stats
    ax.plot(
        range(len(train_list)), train_list,
        color="crimson", ls="--", label="Train"
    )
    ax.plot(
        range(len(valid_list)), valid_list,
        color="#222222", ls=":", label="Valid"
    )

    # set title and labels
    ax.set_title(title, size=20, color="#121212")
    ax.set_xlabel("Epochs", size=14, color="#121212")
    ax.set_ylabel(ylabel, size=14, color="#121212")

    # legend for the plot
    ax.legend(loc=0)

    # grid
    ax.grid()

    if path:
        fig.savefig(path, dpi=600, bbox_inches="tight")

    return fig, ax
