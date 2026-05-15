import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

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


net_i = 10
net_file = f"/root/hdd-data/deep1b/step02/models/deep1b_h{net_i}_model.NN"
time_file = f"/root/hdd-data/deep1b/step02/models/deep1b_h{net_i}_model_time.txt"

for knn in range(10, 101, 10):
    deep1b_test_file = f"/root/hdd-data/deep1b/step02/method_T10/deep1b_k{knn}.txt"
    deep1b_label_file = f"/root/hdd-data/deep1b/step02/method_T10/deep1b_label_h{net_i}_k{knn}.txt"

    test_features = np.loadtxt(deep1b_test_file)
    test_features = torch.from_numpy(test_features).float().to(device=device)

    print(test_features.shape)

    time1 = time.time()

    model = torch.load(net_file)

    data_output = model(test_features)
    data_output = data_output.cpu().detach().numpy()
    np.savetxt(deep1b_label_file, data_output, fmt='%.6f')

    time2 = time.time()
    elapsed_time = time2 - time1
    formatted_time = f"{elapsed_time:.2f}"
    with open(time_file, 'a') as file:
        file.write(str(knn) + ' ' + formatted_time + '\n')

print("trans complete")
