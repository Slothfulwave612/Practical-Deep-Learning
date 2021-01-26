"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module containing utility classes.
"""

# required packages and modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json

import torch
import torch.nn as nn


class DigitDataset:
    """
    class for loading the required dataset.
    """

    def __init__(self, features, target):
        """
        Function to initialise the class object.

        Args:
            features (numpy.ndarray): required feature values.
            target (numpy.ndarray): required target values.
        """
        self.features = features
        self.target = target

    def __len__(self):
        """
        Returns length of the numpy array
        """
        return self.features.shape[0]

    def __getitem__(self, index):
        """
        Function to return an item given by index.
        """
        return {
            'x': torch.tensor(self.features[index, :], dtype=torch.float),
            'y': torch.tensor(self.target[index], dtype=torch.float)
        }


class Engine_ann:
    """
    class to train and evaluate our ANN model.
    """

    def __init__(self, model, optimizer, lr_scheduler, device):
        """
        Function to init the object of the class.

        Args:
            model: PyTorch model.
            optimizer: optimizer function.
            lr_scheduler: learning rate scheduler.
            device: cpu or gpu
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    @staticmethod
    def loss_fn(outputs, targets):
        """
        Function to compute the loss.

        Args:
            outputs: predicted probabilities.
            targets: true-label values.

        Return:
            loss value.
        """
        loss = nn.CrossEntropyLoss()
        targets = targets.type(torch.long)
        return loss(outputs, targets)

    @staticmethod
    def accuracy(targets, outputs):
        """
        Function to compute the loss.

        Args:
            targets: true-label values.
            outputs: predicted probabilities.

        Return:
            accuracy score.
        """
        outputs = outputs.argmax(axis=1)
        return (targets == outputs).sum() / len(targets)

    def train(self, data_loader):
        """
        Function for training.

        Args:
            data_loader (dataloader.DataLoader): contains the required data.

        Returns:
            loss
        """
        self.model.train()

        # init a final loss
        final_loss, final_acc = 0, 0

        for data in data_loader:
            # zero the gradients
            self.optimizer.zero_grad()

            # inputs and targets
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)

            # reshape the inputs
            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)

            # outputs
            outputs = self.model(inputs)

            # calculate loss
            loss = self.loss_fn(outputs, targets)

            # calculate accuracy
            acc = self.accuracy(targets, outputs)

            # calculate accuracy
            self.accuracy(targets, outputs)

            # backward pass
            loss.backward()

            # update the weights
            self.optimizer.step()

            # final loss and accuracy
            final_loss += loss.item()
            final_acc += acc.item()
        
        print(self.optimizer.state_dict()["param_groups"][0]["lr"])

        return round(final_loss / len(data_loader), 3), \
            round(final_acc / len(data_loader), 3)

    def evaluate(self, data_loader, save_results=False):
        """
        Function for evaluation.

        Args:
            data_loader (dataloader.DataLoader): contains the required data.
            save_results (bool, optional): to save wrongly classified images.

        Returns:
            loss
        """
        self.model.eval()

        # init loss and accuracy
        final_loss, final_acc = 0, 0

        # init an empty list
        if save_results:
            results = []

        for data in data_loader:
            # inputs and targets
            inputs = data['x'].to(self.device)
            targets = data['y'].to(self.device)

            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)

            # outputs
            outputs = self.model(inputs)

            # calculate loss
            loss = self.loss_fn(outputs, targets)

            # calculate accuracy
            acc = self.accuracy(targets, outputs)

            # add to final loss and accuracy
            final_loss += loss.item()
            final_acc += acc.item()

            if save_results:
                # get predicted labels
                outputs = outputs.argmax(axis=1)

                # get index where images are wrongly classified
                index = np.where(outputs.cpu() != targets.cpu())[0]

                # transfer inputs to cpu
                inputs = inputs.cpu()

                for i in range(len(index)):
                    # get pixel value of the image
                    image = np.array(inputs[index[i]]).tolist()

                    # get target value
                    target = int(targets[index[i]].item())

                    # get predicted value
                    pred_val = outputs[index[i]].item()

                    results.append({
                        "image": image, "target": target,
                        "prediction": pred_val
                    })

        # save the file as json
        if save_results:
            with open("data/wrong_pred.json", "w") as json_file:
                json.dump(results, json_file)
            
        self.lr_scheduler.step(final_loss / len(data_loader))

        return round(final_loss / len(data_loader), 3), \
            round(final_acc / len(data_loader), 3)


class Model(nn.Module):
    """
    class for defining NN architecture.
    """

    def __init__(self):
        # to access methods of parent class
        super(Model, self).__init__()

        # add convolutional layer
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.Flatten(start_dim=1),
            nn.Linear(
                in_features=256*3*3, out_features=256
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(
                in_features=256, out_features=256
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(
                in_features=256, out_features=10
            )
        )

        self.initialize_weights()

    def forward(self, x):
        """
        Function to perform forward pass.

        Args:
            x: data.
        """
        return self.model(x)

    def initialize_weights(self):
        """
        Function to initialize weights for
        Linear and Conv2d layers.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


def create_line_plot(
    ax, train_list, valid_list, xlabel, ylabel, title
):
    """
    Function to create a line plot.

    Args:
        ax (axes.Axes): axes object.
        train_list (list): containing train data.
        valid_list (list): containing valid data.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        title (str): title of the plot
    """
    # default font family
    rcParams["font.family"] = "serif"

    # set facecolor
    ax.set_facecolor("#EEE6CC")

    # plot train data
    ax.plot(
        range(len(train_list)), train_list,
        color="#8142E9", ls="--", label="Train"
    )

    # plot valid data
    ax.plot(
        range(len(valid_list)), valid_list,
        color="#222222", ls=":", label="Valid"
    )

    # set title and labels
    ax.set_title(title, size=20, color="#222222")
    ax.set_xlabel(xlabel, size=14, color="#222222")
    ax.set_ylabel(ylabel, size=14, color="#222222")

    # legend for the plot
    ax.legend(loc=0)

    # grid
    ax.grid(alpha=0.5)

    # hide spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_over_epochs(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to plot line plot for train
    accuracy/loss and test accuracy/loss.

    Args:
        array: containing accuracy/loss values for train and valid set.

    Returns:
        figure.Figure: figure object.
        axes.Axes: axes obejct.
    """
    # create subplot
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        facecolor="#EEE6CC", figsize=(14, 8), dpi=600
    )

    # plot for accuracy
    create_line_plot(
        axes[0], train_acc, valid_acc,
        "Epochs", "Accuracy", "Accuracy: Train and Valid Set"
    )

    # plot for loss
    create_line_plot(
        axes[1], train_loss, valid_loss,
        "Epochs", "Loss", "Loss: Train and Valid Set"
    )

    return fig, axes


# def foo(m):
#     if type(m) == nn.Conv2d:
#         print(type(m))

# model = Model()

# model.apply(foo)
