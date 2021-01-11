"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module for training an Artificial Neural Network.
"""

# import necessary packages/modules
import numpy as np
import pandas as pd

import torch

from . import utils_class, utils_func

# defualt values
DEVICE = "cuda"
EPOCHS = 50
BATCH_SIZE = 350


def run_training(target, save_model=False):
    """
    Function to train ANN.

    Args:
        target (str): label column name.
        save_model (bool, optional): to save the model. Defaults to False.
    """

    # load the dataset
    train_df = pd.read_pickle("data/train.pkl")
    valid_df = pd.read_pickle("data/valid.pkl")
    test_df = pd.read_pickle("data/test.pkl")

    print(train_df.shape, valid_df.shape, test_df.shape)

    # split for training data
    X_train = train_df.drop(target, axis=1).to_numpy()
    y_train = train_df[target].to_numpy()

    # split for validation data
    X_valid = valid_df.drop(target, axis=1).to_numpy()
    y_valid = valid_df[target].to_numpy()

    # split for test data
    X_test = test_df.drop(target, axis=1).to_numpy()
    y_test = test_df[target].to_numpy()

    # init object of DigitDataset for train data-set
    train_dataset = utils_class.DigitDataset(
        features=X_train, target=y_train
    )

    # init object of DigitDataset for valid data-set
    valid_dataset = utils_class.DigitDataset(
        features=X_valid, target=y_valid
    )

    # init object of DigitDataset for valid data-set
    test_dataset = utils_class.DigitDataset(
        features=X_test, target=y_test
    )

    # create data loaders for train data set
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # create data loaders for valid data set
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE
    )

    # create data loaders for test data set
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE
    )

    # init object to make model
    model = utils_class.Model(
        num_features=X_train.shape[1],
        num_targets=10,
        num_layers=7,
        hidden_size=70,
        dropout=0.05
    )

    # transfer to GPU
    model = model.to(DEVICE)

    # make an optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-8)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=EPOCHS
    )

    # init object of Engine class
    engine = utils_class.Engine_ann(
        model, optimizer, lr_scheduler, DEVICE
    )

    # init empty list for loss and accuracy
    loss_list_train, acc_list_train = [], []
    loss_list_valid, acc_list_valid = [], []

    for epoch in range(1, EPOCHS + 1):
        if epoch == EPOCHS:
            save_results = True
        else:
            save_results = False

        # train and valid loss and accuracy
        train_loss, train_acc = engine.train(train_loader)
        valid_loss, valid_acc = engine.evaluate(valid_loader, save_results)

        # append the info
        loss_list_train.append(train_loss)
        loss_list_valid.append(valid_loss)
        acc_list_train.append(train_acc)
        acc_list_valid.append(valid_acc)

        print(f"Epoch: {epoch}:")
        print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}")
        print(f"Train Acc: {train_acc}, Valid Acc: {valid_acc}")
        print()

    # test loss and accuracy
    test_loss, test_acc = engine.evaluate(test_loader)
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")

    fig, ax = utils_func.plot_over_epochs(
        acc_list_train, acc_list_valid,
        "Accuracy Results", "Accuracy",
        "plots/ann_accuracy.jpg"
    )

    fig, ax = utils_func.plot_over_epochs(
        loss_list_train, loss_list_valid,
        "Loss Results", "Loss",
        "plots/ann_loss.jpg"
    )

    torch.save(model, f"models/model_ann_final.pt")


if __name__ == "__main__":
    torch.manual_seed(42)
    run_training("label", True)
