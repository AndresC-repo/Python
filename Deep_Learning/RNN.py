import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
from tqdm import tqdm
# Set seed
seed=0
torch.manual_seed(seed)
random.seed(seed)

# -------------------------------------------- #
#      RNN
# -------------------------------------------- #

class RNN(nn.Module):
    # define model: init and forward
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()  # calls initialization method of parent class (nn.Module)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_lenght, num_classes)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # hidden state init
        # forward pass
        out, _ = self.rnn(x, h0)  # _ is the hidden state
        out = out.reshape(out.shape[0], -1)  # batch, flatten all else
        out = self.fc(out)
        return out

# -------------------------------------------- #
#       Training
# -------------------------------------------- #

def train_network(train_loader, model):
    # loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training:
    for ep in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

            data = data.to(device=device).squeeze(1)  # from Nx1x28x28 to Nx28x28
            targets = targets.to(device=device)

            # Forward Pass
            logits = model(data)
            loss = criterion(logits, targets)

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# -------------------------------------------- #
#   Main
# -------------------------------------------- #
if __name__ == "__main__":
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------- #
    #      Hyperparameters
    # -------------------------------------------- #
    '''  Nx1x28x28 -> which can be seen as 28 time
         sequences with 28 features '''

    input_size = 28
    sequence_lenght = 28
    num_layers = 2
    hidden_size = 256
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 3


    # FC no normalization
    normalize = True
    train_loader, test_loader = load_data(normalize)  # Get data
    model = RNN(input_size=input_size, hidden_size=hidden_size, \
        num_layers=num_layers, num_classes=num_classes).to(device)  # Initialize network
    model = train_network(train_loader, model)  # Train
    check_acc(train_loader, model)
    check_acc(test_loader, model)


# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #
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
            # x shaped [N, 1, 28, 28]
            x = x.to(device=device).squeeze(1)  # RNN takes Nx28x28
            # y shaped [N]
            y = y.to(device=device)
            # gets logits Nx10
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


