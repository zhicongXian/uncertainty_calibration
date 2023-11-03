# This is an example Python script on how to use the numpy implementation on temperature scaling, here we use the
# same example from scikit-learn library https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html#sphx-glr-auto-examples-calibration-plot-calibration-multiclass-py

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD Style.

import numpy as np
from sklearn.datasets import make_blobs
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from temperature_scaling_calibrator_np import TemperatureScalingCalibrator
from sklearn.preprocessing import OneHotEncoder

np.random.seed(0)
nb_classes = 3
n_features = 20
# 1. prepare data
X, y = make_blobs(
    n_samples=2000, n_features=n_features, centers=nb_classes, random_state=42, cluster_std=5.0
)
enc = OneHotEncoder()
y = enc.fit_transform(y[:, None]).todense()
X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:1000], y[600:1000]
X_train_valid, y_train_valid = X[:1000], y[:1000]
X_test, y_test = X[1000:], y[1000:]

# 2. fit a classifier
# transform to torch tensor
tensor_x_train = torch.Tensor(X_train[:, None])  # for pytorch, channel dimension comes first
tensor_y_train = torch.Tensor(y_train)
tensor_x_valid = torch.Tensor(X_valid[:, None])
tensor_y_valid = torch.Tensor(y_valid)

train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create training dataset
train_dataloader = DataLoader(train_dataset, batch_size=32)  # create training dataloader

valid_dataset = TensorDataset(tensor_x_valid, tensor_y_valid)  # create training dataset
valid_dataloader = DataLoader(valid_dataset, batch_size=32)  # create training dataloader


# define a simple classfier


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)  # feature_map_dim: [n_features - 5 + 1] = [16]
        self.pool = nn.MaxPool1d(2, 2)  # [(16 - 2)/2 + 1] = [8]
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nb_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # feature_dim: 8
        x = self.pool(
            F.relu(self.conv2(x)))  # feature_dim: 8 - 5 + 1 = 4 followed by max pooling yields (4-2)/2 + 1 = 2
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


net = Classifier()

# define a loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train the network:
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Test and calibrate the data:
predicted_probabilities = []
ground_truth_labels = []
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in valid_dataloader:
        features, labels = data
        # calculate outputs by running images through the network
        outputs = net(features)
        # the class with the highest energy is what we choose as prediction
        predicted_probabilities.append(outputs.detach().numpy())
        ground_truth_labels.append(labels.detach().numpy())

# start calibration

predicted_probabilities_np = np.concatenate(predicted_probabilities, 0)
ground_truth_labels_np = np.concatenate(ground_truth_labels, 0)

calibrator = TemperatureScalingCalibrator()

calibrated_prob = calibrator.fit_predict(predicted_probabilities_np, ground_truth_labels_np,
                                         verbose=True)

