"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module containing utility functions.
"""

# required packages and modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn import metrics

import torch
from torch import nn

from tqdm import tqdm


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


def model_training_ann(
    model, train_df, label, valid_df=None,
    epochs=500, lr=0.001, use_gpu=False
):
    """
    Function to train the model for ANN.

    Args:
        model: PyTorch Model.
        train_df (pandas.DataFrame): train-data values.
        label (str): lable column name.
        valid_df (pandas.DataFrame, optional): valid-data values.
                                               Defaults to None.
        epochs (int, optional): number of iterations. Defaults to 500.
        lr (float, optional): learning rate. Defaults to 0.01.
        use_gpu (bool, optional): to use GPU or not. Defaults to False.
    """
    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    # split train data to X and y
    X_train = torch.tensor(
        train_df.drop(label, axis=1).values
    )
    y_train = torch.tensor(
        train_df[label].values
    )

    if valid_df is not None:
        # split valid data to X and y
        X_valid = torch.tensor(
            valid_df.drop(label, axis=1).values
        )
        y_valid = torch.tensor(
            valid_df[label].values
        )

    if use_gpu:

        # check for GPU
        if torch.cuda.is_available():
            print(f"System has {torch.cuda.get_device_name()}")
            print()
            print("Transfering Training Data to GPU")

            # transfer training data to GPU
            X_train = X_train.cuda()
            y_train = y_train.cuda()

            if valid_df is not None:
                print()
                print("Transfering Validation Data to GPU")

                # transfer validation data to GPU
                X_valid = X_valid.cuda()
                y_valid = y_valid.cuda()

            # transfer model architecture to GPU
            model = model.cuda()

        else:
            print("GPU not available")

    # train the model
    for i in range(epochs):

        # forward pass --> generate predictions
        y_pred_train = model.forward_pass(X_train.float())

        # convert y_pred_train from probability to class-labels
        y_pred_train = y_pred_train.argmax(axis=1)

        # calculate loss on training data
        print(y_pred_train)
        print(y_train)
        train_loss = loss_fn(y_pred_train, y_train)

        if valid_df is not None:

            # generate predictions on validation data
            y_pred_valid = model.forward_pass(X_valid.float())

            # convert y_pred_valid from probability to class-labels
            y_pred_valid = y_pred_valid.argmax(axis=1)

        if i % 101 == 0:

            # calculate accuracy
            train_acc = metrics.accuracy_score(
                y_train, y_pred_train
            )

            if valid_df is not None:
                # valid accuracy
                valid_acc = metrics.accuracy_score(
                    y_valid, y_pred_valid
                )

                # valid loss
                valid_loss = loss_fn(y_pred_valid, y_valid)

            print()
            print(f"Epochs: {i}\nTraining Loss: {train_loss} | \
                Training Acc: {train_acc}")

            if valid_df:
                print(f"Validation Loss: {valid_loss} | \
                    Validation Acc: {valid_acc}")

        # zero the gradients
        optimizer.zero_grad()

        # backward pass
        train_loss.backward()

        # update weights
        optimizer.step()
