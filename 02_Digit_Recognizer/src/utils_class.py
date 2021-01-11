"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module containing utility classes.
"""

# required packages and modules
import numpy as np
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
            features (torch.tensor): required feature values.
            target (torch.tensor): required target values.
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

        self.lr_scheduler.step()

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

        return round(final_loss / len(data_loader), 3), \
            round(final_acc / len(data_loader), 3)


class Model(nn.Module):
    """
    class for defining NN architecture.
    """

    def __init__(
        self, num_features, num_targets, num_layers, hidden_size, dropout
    ):
        """
        Function to init object of class.

        Args:
            num_features (int): total number of input features.
            num_targets (int): total number of targets.
            num_layers (int): total number of hidden-layers.
            hidden_size (int): total number of neurons in each hidden-layers.
            dropout (float): dropout value.
        """
        # to access methods of parent class
        super(Model, self).__init__()

        # init empty list
        layers = []

        # append input layer
        layers.append(
            nn.Linear(
                in_features=num_features, out_features=hidden_size,
                bias=False
            )
        )
        torch.nn.init.xavier_normal_(layers[-1].weight)
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.SELU())

        # append hidden layers
        for _ in range(num_layers):
            layers.append(
                nn.Linear(
                    in_features=hidden_size, out_features=hidden_size,
                    bias=False
                )
            )
            torch.nn.init.xavier_normal_(layers[-1].weight)
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.SELU())

        # append output layer
        layers.append(
            nn.Linear(
                in_features=hidden_size, out_features=num_targets
            )
        )
        torch.nn.init.xavier_normal_(layers[-1].weight)
        # layers.append(
        #     nn.Softmax(dim=1)
        # )

        # make model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Function to perform forward pass.

        Args:
            x: data.
        """
        return self.model(x)
