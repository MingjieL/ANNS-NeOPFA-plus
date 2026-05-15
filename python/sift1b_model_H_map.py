import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()
N_HIDDEN = 4  # the number of hidden layers


class OPHashNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, batch_normalization):
        super(OPHashNet, self).__init__()

        self.fc0 = torch.nn.Linear(n_feature, n_hidden)
        self.set_init(self.fc0)

        self.fc1 = torch.nn.Linear(n_hidden, n_hidden)
        self.set_init(self.fc1)

        self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
        self.set_init(self.fc2)

        self.fc3 = torch.nn.Linear(n_hidden, n_hidden)
        self.set_init(self.fc3)

        self.output = torch.nn.Linear(n_hidden, n_output)
        self.set_init(self.output)

    def set_init(self, layer):
        torch.nn.init.normal_(layer.weight)
        layer.weight.data = F.normalize(layer.weight.data)
        torch.nn.init.constant_(layer.bias, 0.001)

    def forward(self, x):
        x1 = self.fc0(x)
        x1 = F.elu(x1)

        x2 = self.fc1(x1)
        x2 = F.elu(x2)

        x3 = self.fc2(x2)
        x3 = F.elu(x3)

        x4 = self.fc3(x3)
        x4 = F.elu(x4)

        x5 = self.output(x4)
        return x5


if __name__ == "__main__":
    data_name = "sift1b"
    v = 10
    v_v = 38
    net_i = 5
    n_hidden = 512
    n_output = 25
    data_file = "/root/hdd-data/sift1b/data.ds"
    query_file = "/root/hdd-data/sift1b/query.txt"
    net_folder = "/root/hdd-data/sift1b/Index_Learning_code/models_sift1b_d{}_v{}_{}".format(n_output, v, v_v)
    n_num = 1000000000
    q_num = 100
    sub_num = 100000
    n_dimensions = 128
    batch_normalization = False
    d_file = "{}/{}_base_d{}_v{}_vv{}_{}.bin".format(net_folder, data_name, n_output, v, v_v, net_i)
    q_file = "{}/{}_query_d{}_v{}_vv{}_{}.txt".format(net_folder, data_name, n_output, v, v_v, net_i)
    # model_file = net_folder + "/sift1b_last_iteration.NN"
    model_file = net_folder + "/sift1b_{}.NN".format(net_i)

    opHashNet = OPHashNet(n_dimensions, n_hidden, n_output, batch_normalization)
    state_dict = torch.load(model_file)
    fixed_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    opHashNet.load_state_dict(fixed_state_dict)
    opHashNet.to('cuda')
    print(opHashNet)

    cache = np.zeros((sub_num, n_dimensions), dtype=np.float32)

    with open(data_file, 'rb') as fps, open(d_file, 'wb') as ofs:
        for i in range(n_num // sub_num):
            fps.readinto(cache.view(dtype=np.float32))
            data = np.array(cache)

            # mean_norm = ((data ** 2).sum(-1) ** 0.5).mean().item()
            # normalize = lambda v: v / mean_norm
            # data = np.array(list(map(normalize, data)))
            # data = torch.from_numpy(data).float().to(device=device)

            mu = np.mean(data)
            sigma = np.std(data)
            z_normalize = lambda v: (v - mu) / sigma
            data = np.array(list(map(z_normalize, data)))
            data = torch.from_numpy(data).float().to(device=device)

            data_output = opHashNet(data)
            data_output_cpu = data_output.cpu().detach().numpy()

            for j in range(sub_num):
                ofs.write(data_output_cpu[j].tobytes())
            if i % 100 == 0:
                print(f"already finish {i // 100 + 1}%")
    fps.close()
    ofs.close()

    query_data = np.loadtxt(query_file)

    # mean_norm = ((query_data ** 2).sum(-1) ** 0.5).mean().item()
    # normalize = lambda v: v / mean_norm
    # query_data = np.array(list(map(normalize, query_data)))

    mu = np.mean(query_data)
    sigma = np.std(query_data)
    z_normalize = lambda v: (v - mu) / sigma
    query_data = np.array(list(map(z_normalize, query_data)))

    query_data_output = opHashNet(torch.from_numpy(query_data).float().to(device=device))
    query_output_cpu = query_data_output.cpu().detach().numpy()

    np.savetxt(q_file, query_output_cpu, fmt='%.6f')
