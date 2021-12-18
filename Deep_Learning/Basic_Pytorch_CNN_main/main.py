"""
Simple main execution file for a CNN or FC
created with Pytorch

Normalization
Accuracy as metric
"""

# Imports
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# -------------------------------------------- #
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# -------------------------------------------- #
import time
# -------------------------------------------- #
# Set seed
seed=0
torch.manual_seed(seed)
random.seed(seed)


# -------------------------------------------- #
#     CONV NN
# -------------------------------------------- #

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# -------------------------------------------- #
#      FC
# -------------------------------------------- #

class NN(nn.Module):

    # define model: init and forward
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()  # calls initialization method of parent class (nn.Module)
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------------------- #
# Data loader
# -------------------------------------------- #


def load_data(normalize):
    if normalize:
        transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        print(f'\n with normalization')
    else:
        transformation = transforms.ToTensor()
        print(f'\n NO normalization')


    # Load data: Datasetclass and Dataloader
    train_dataset = datasets.MNIST(root='dataset/', train=True,transform=transformation, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    if not normalize:
        print(f'\n mean and std: {get_mean_std(train_loader)}')
    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transformation, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# -------------------------------------------- #
#       Training
# -------------------------------------------- #

def train_network(train_loader, model, FC=True):
    # loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training:
    for ep in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

            data = data.to(device=device)
            targets = targets.to(device=device)

            # Reshape to flatten
            if FC:
                data = data.reshape(data.shape[0], -1)

            # Forward Pass
            logits = model(data)
            loss = criterion(logits, targets)

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

# -------------------------------------------- #
# Metrics: Accuracy
# -------------------------------------------- #
def check_acc(loader, model, FC=True):

    # Check accuracy in training
    if loader.dataset.train:
        print('checking acc on training data')
    else:
        print('checking acc on test data')
    num_correct = 0
    num_samples = 0
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for x,y in loader:
            # x shaped [64, 1, 28, 28]
            x = x.to(device=device)
            # y shaped [64]
            y = y.to(device=device)
            if FC:
                # Reshape to flatten from [64, 1, 28, 28] to [64, 784]
                x = x.reshape(x.shape[0], -1)
            # gets logits 64x10
            logits = model(x)
            _, prediction = logits.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)
        print(f'Got {num_correct} / {num_samples} with acc {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
    end_time = time.time()
    print(f'total time is: {end_time-start_time:.2f}')

# -------------------------------------------- #
#   Normalization: Mean and Std
# -------------------------------------------- #
# This is done for every channel individually
# Batch_size, Channels, H, W -> dim=[0, 2, 3] 
# Std[X] = (E[X^2] - E[X]^2)^0.5

def get_mean_std(loader):
    c_sum = 0  # E[X^2]
    c_sqrd_sum = 0  # E[X]^2
    num_batches = 0  # Make it flexible

    for data, _ in tqdm(loader):
        
        c_sum += torch.mean(data, dim=[0, 2, 3])
        c_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = c_sum / num_batches
    std = (c_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# -------------------------------------------- #
#   Main
# -------------------------------------------- #
if __name__ == "__main__":
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameteres: im_size, num_classes, lr, batch_size, max_epochs
    input_size = 784  # 28x28
    in_channels = 1
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 1


    ## test model initialization: wanted (batch_size, num_classes)
    # model = NN(28*28, 10)
    # x = torch.rand(64, 28*28)  # 64 is the mini_batch size
    # print(model(x).shape)


    # FC no normalization
    normalize = False
    train_loader, test_loader = load_data(normalize)  # Get data
    model = NN(input_size=input_size, num_classes=num_classes).to(device)  # Initialize network
    model = train_network(train_loader, model)  # Train
    check_acc(train_loader, model)
    check_acc(test_loader, model)

    # FC w normalization
    normalize=True
    train_loader, test_loader = load_data(normalize)  # Get data
    model = NN(input_size=input_size, num_classes=num_classes).to(device)  # Initialize network
    model = train_network(train_loader, model)  # Train
    check_acc(train_loader, model)
    check_acc(test_loader, model)

    # CNN no normalization
    normalize=False
    train_loader, test_loader = load_data(normalize)  # Get data
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)  # Initialize network
    model = train_network(train_loader, model, FC=False)  # Train
    check_acc(train_loader, model, FC=False)
    check_acc(test_loader, model, FC=False)

    # CNN w normalization
    normalize=True
    train_loader, test_loader = load_data(normalize)  # Get data
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)  # Initialize network
    model = train_network(train_loader, model, FC=False)  # Train
    check_acc(train_loader, model, FC=False)
    check_acc(test_loader, model, FC=False)
