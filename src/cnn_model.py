import os
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

# FRO GRAD - CAM
import torchvision.transforms.functional as TF
from PIL import Image

import utils_2d as utils


class ICModel(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=2)
        self.conv2 = nn.Conv2d(8, 24, kernel_size=4, stride=2)

        # FULLY CONNECTED LAYERS
        self.fc1 = nn.Linear(136752, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 3)

    def forward(self, x):
        # CONV - 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # CONV - 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        flattened_size = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]

        x = x.view(-1, flattened_size)
        # FULLY CONNECTED LAYERS
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.out(x)

        return F.log_softmax(x, dim=1)

    def train(self, dataset_dir='', epochs=5, batch_size=16, seed=35, learning_rate=0.001, model_weights_path=''):
        if dataset_dir == '':
            raise Exception("Please enter a valid dataset directory path!")

        train_correct = []
        train_losses = []

        torch.manual_seed(seed)

        # CRITERION AND OPTIMIZER SETUP
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        optim_width, optim_height = 1250, 1206

        data_transforms = transforms.Compose([
            transforms.Resize((optim_width, optim_height)),  # Resize images to average dimensions
            transforms.Grayscale(),
            transforms.RandomRotation(degrees=20),  # Rotate images randomly up to 20 degrees
            transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally with a probability of 0.5
            transforms.RandomVerticalFlip(p=0.5),  # Flip images vertically with a probability of 0.5
            transforms.ColorJitter(brightness=0.2,  # Adjust brightness with a factor of 0.2
                                   contrast=0.2,  # Adjust contrast with a factor of 0.2
                                   saturation=0.2,  # Adjust saturation with a factor of 0.2
                                   hue=0.1),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.456], std=[0.456])  # Normalize images
        ])

        dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            trn_corr = 0

            for b, (X_train, y_train) in enumerate(train_loader):
                b += 1
                y_pred = self(X_train)
                loss = criterion(y_pred, y_train)

                predicted = torch.max(y_pred, dim=1)[1]
                batch_corr = (predicted == y_train).sum()

                trn_corr += batch_corr.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if b % 4 == 0:
                    print(f'Epoch: {epoch}  Batch: {b}  Loss: {loss.item()}')

        train_losses.append(loss)
        train_correct.append(trn_corr)

        if (model_weights_path != '') & os.path.exists(model_weights_path) & os.path.isdir(model_weights_path):
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_weights_path)

    def test(self, dataset_dir='', batch_size=16):
        if dataset_dir == '':
            raise Exception("Please enter a valid dataset directory path!")

        optim_width, optim_height = utils.calcAvrImgSize(dataset_dir)
        test_losses = []
        tst_crr = 0

        criterion = nn.CrossEntropyLoss()

        data_transforms = transforms.Compose([
            transforms.Resize((optim_width, optim_height)),  # Resize images to average dimensions
            transforms.Grayscale(),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.456], std=[0.456])  # Normalize images
        ])

        dataset = datasets.ImageFolder(root=dataset_dir, transform=data_transforms)
        test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                y_val = self(X_test)
                predicted = torch.max(y_val.data, dim=1)[1]
                tst_crr += (predicted == y_test).sum()

            loss = criterion(y_val, y_test)
            test_losses.append(loss.item())

            test_results = {
                'true_positive': tst_crr,
                'false_positive': len(dataset.imgs) - tst_crr
            }

        return test_results, test_losses
