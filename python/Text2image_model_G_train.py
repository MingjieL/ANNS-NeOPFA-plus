import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.input = nn.Linear(input_size, 512)
        self.set_init(self.input)
        self.hidden1 = nn.Linear(512, 512)
        self.set_init(self.hidden1)
        self.hidden2 = nn.Linear(512, 512)
        self.set_init(self.hidden2)
        self.hidden3 = nn.Linear(512, 512)
        self.set_init(self.hidden3)
        self.output = nn.Linear(512, output_size)
        self.set_init(self.output)

    def set_init(self, layer):
        torch.nn.init.normal_(layer.weight)
        layer.weight.data = F.normalize(layer.weight.data)
        torch.nn.init.constant_(layer.bias, 0.001)

    def forward(self, x):
        x = F.elu(self.input(x))
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        x = F.elu(self.hidden3(x))
        y = self.output(x)
        return y


Text2image_train_file = "/root/hdd-data/Text2image/step02/train/Text2image_train_10M.txt"
Text2image_label_file = "/root/hdd-data/Text2image/step02/train/Text2image_label_10M.txt"
net_file = "/root/hdd-data/Text2image/step02/models/Text2image_h10_model.NN"

train_features = np.loadtxt(Text2image_train_file)
train_features = torch.from_numpy(train_features).float()

train_labels = np.loadtxt(Text2image_label_file)
train_labels = torch.from_numpy(train_labels).float()
train_labels = 100.0 * train_labels

dataset = TensorDataset(train_features, train_labels)
batch_size = 64
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SimpleNet(train_features.shape[1], train_labels.shape[1])

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.to(device)

for epoch in range(1, 10):
    for batch_idx, (features, labels) in enumerate(data_loader):
        features, labels = features.to(device), labels.to(device)
        output = model(features)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch: [{epoch}], Batch: [{batch_idx + 1}/{len(data_loader)}] ,Loss: {loss.item():.4f}")

torch.save(model, net_file)

print("train complete")
