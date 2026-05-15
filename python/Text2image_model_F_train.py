import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_accuracy = 0.0


class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes=64):
        super(SimpleNet, self).__init__()
        self.input = nn.Linear(input_size, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.input(x))
        y = self.output(x)
        return y


def train(model, train_loader, optimizer, epoch):
    time1 = time.time()
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)
        target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    average_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    time2 = time.time()
    print(f'Epoch {epoch} finished with average loss {average_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {time2-time1:.2f}s')


def test(model, test_loader, epoch, net_folder, topk):
    model.eval()
    global best_accuracy
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            total_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            pred_list = pred.squeeze().tolist()
            output_file = "{}/cluster_k{}_v{}.txt".format(net_folder, topk, epoch)
            with open(output_file, 'w') as f:
                for index in pred_list:
                    f.write(str(index) + '\n')

            correct += pred.eq(target.view_as(pred)).sum().item()

            average_loss = total_loss / len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                print(pred.reshape(1, -1))
            print(f'Test set: Average loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
            return average_loss, accuracy


Text2image_query_train_file = "/root/hdd-data/Text2image/cluster/train/train_query_100K.txt"
Text2image_query_test_file = "/root/hdd-data/Text2image/cluster/test/query_test.txt"

Text2image_cluster_train_file = "/root/hdd-data/Text2image/cluster/train/new_cluster_k100.txt"
Text2image_cluster_test_file = "/root/hdd-data/Text2image/cluster/test/cluster_k100.txt"

net_folder = "/root/hdd-data/Text2image/cluster/models"

train_features = np.loadtxt(Text2image_query_train_file)
train_features = torch.from_numpy(train_features).float()

train_labels = np.loadtxt(Text2image_cluster_train_file)
train_labels = torch.from_numpy(train_labels).float()

test_features = np.loadtxt(Text2image_query_test_file)
test_features = torch.from_numpy(test_features).float()
test_labels = np.loadtxt(Text2image_cluster_test_file)
test_labels = torch.from_numpy(test_labels).long()

train_data = torch.utils.data.TensorDataset(train_features, train_labels)
test_data = torch.utils.data.TensorDataset(test_features, test_labels)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

model = SimpleNet(input_size=200)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

for epoch in range(1, 1001):
    train(model, train_loader, optimizer, epoch)

    net_file = "{}/{}.NN".format(net_folder, epoch)
    torch.save(model, net_file)

    test(model, test_loader, epoch, net_folder, 100)

print("finish training")
