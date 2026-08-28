import numpy as np
from torch import device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def dataset_load():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )



def check_Gpu():
    if(torch.cuda.is_available()):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
        model = MLP().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return device, criterion, optimizer, model
    return None

def training_data(device, criterion, optimizer, model, train_loader):
    for epoch in range(5):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()

def accuracy(model, test_loader):
    model.eval()
    correct,total = 0,0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == Y_batch).sum().item()
            total += Y_batch.size(0)

def print_hi(name):
    print(f'Hi, {name}')



if __name__ == '__main__':
    print_hi('PyCharm')


