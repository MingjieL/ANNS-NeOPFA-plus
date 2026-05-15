import torch
import numpy as np
import torch.nn.functional as F
import random
import time
import os
import struct
from torch.utils.tensorboard import SummaryWriter
from heapq import heappush, heappop, nlargest, nsmallest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()
N_HIDDEN = 4  # the number of hidden layers


def gt_calculator2(data, queries):
    gt, gt_dist = [], []
    for query in queries:
        matrix1 = (query - data) ** 2
        arr1 = matrix1.sum(1)
        dists, ids = torch.sort(arr1)
        gt.append(ids.data.cpu().numpy())
        gt_dist.append(dists.data.cpu().numpy())
    # return torch.tensor(gt)
    return gt, gt_dist


def write_float(data, file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w') as file:
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                write_str = '%f ' % (data[x, y])
                file.write(write_str)
            file.write('\n')


def write_binary(data, file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'wb') as bin_file:
        for row in data:
            for value in row:
                binary_data = struct.pack('f', value)
                bin_file.write(binary_data)


def write_para1(data, file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'a+') as file:
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                write_str = '%f ' % (data[x, y])
                file.write(write_str)
            file.write('\n')


def write_para2(data, file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'a+') as file:
        for x in range(len(data)):
            file.write('%f ' % (data[x]))
            if x != 0 and x % 10 == 0:
                file.write('\n')


def get_distance_single(vector1, vector2):
    return ((vector1 - vector2) ** 2).sum()


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


def neopfa(train_data, train_query, test_data, test_query, kmeans_partitions, center_vectors,
           k, rerank_t, verify_t, n_kmeans, n_kmeans_selected, test_gt,
           n_hidden, n_output, topK, S_, lr,
           epochs, batch_size, batch_normalization,
           log_file, log_loss, log_loss2, log_real1, log_real2, log_recall, log_lr, net_folder, st, alpha):
    n_points = int(train_data.shape[0])
    n_queries = int(train_query.shape[0])
    n_dim = int(train_data.shape[1])
    b = 1
    IDs = [i for i in range(n_queries)]
    k2 = topK + S_

    # opHashNet = OPHashNet(n_dim, n_hidden, n_output, batch_normalization).to(device=device)
    opHashNet = OPHashNet(n_dim, n_hidden, n_output, batch_normalization)
    # opHashNet = trained_net
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        opHashNet = torch.nn.DataParallel(opHashNet, device_ids=[0, 1, 2, 3])
    opHashNet.to(device=device)
    print(opHashNet)
    optimizer = torch.optim.Adam(opHashNet.parameters(), lr=lr)

    data = torch.from_numpy(train_data).float().to(device=device)
    train_query = torch.from_numpy(train_query).float().to(device=device)
    test_data = torch.from_numpy(test_data).float()
    n_points_test = test_data.shape[0]
    test_query = torch.from_numpy(test_query).float()
    test_gt = torch.from_numpy(test_gt).float()
    writer = SummaryWriter(log_file)

    # training
    iter1, net_i, r_i, d_i, para_i, recall_i, recall = 0, 0, 0, 0, 0, 0, 0
    gt_rank1 = np.arange(1, topK + 1)
    tt = torch.from_numpy(np.linspace(1, 10, topK)).float().to(device=device)
    best_recall = 0

    for epoch_i in range(epochs):
        np.random.shuffle(IDs)
        step_i = 0
        for step in range(0, n_queries, batch_size):
            iter1 += 1
            step_i += 1

            if iter1 % 14000 == 0:
                net_file = "{}/{}_{}.NN".format(net_folder, data_name, net_i)
                net_i += 1
                torch.save(opHashNet, net_file)
            if iter1 % 2000 == 0:
                recall_file = "{}/{}_recall.txt".format(net_folder, data_name)
                # test the real performance
                with torch.no_grad():
                    sorted_lists_all = []
                    for nk in range(n_kmeans):
                        data_embeddings = opHashNet(test_data[kmeans_partitions[nk]].to(device=device))
                        # data_embeddings_cpu = data_embeddings.data.cpu().numpy()
                        sorted_lists = build_index(data_embeddings.data.cpu().numpy(), kmeans_partitions[nk])
                        sorted_lists_all.append(sorted_lists)
                    # opHashNet_cpu = opHashNet.to("cpu")
                    print("finished constructing/loading index....")
                    query_embeddings = opHashNet(test_query.to(device=device))
                    data_embeddings_all = opHashNet(test_data.to(device=device))
                    best_ids = search(n_points_test, data_embeddings_all.data.cpu().numpy(),
                                      test_query, center_vectors,
                                      sorted_lists_all, query_embeddings.data.cpu().numpy(),
                                      k, rerank_t, verify_t)
                    recall = get_recall(best_ids, test_gt)  # compute recall
                    if recall > best_recall:
                        best_recall = recall
                    writer.add_scalar(log_recall, recall, recall_i)
                    directory = os.path.dirname(recall_file)
                    os.makedirs(directory, exist_ok=True)
                    with open(recall_file, 'a+') as file:
                        file.write("%0.6f\n" % recall)

                    recall_i += 1

            time1 = time.time()

            ids = IDs[step: step + batch_size]
            queries = train_query[ids]

            ## compute the gt_id and gt_dist
            gt, gt_dist = gt_calculator2(data, queries)  # compute the ground truth
            gt = np.array(gt)
            gt_dist = np.array(gt_dist)
            time2 = time.time()

            relevance = np.exp(-gt_dist[0] / gt_dist[0, n_points - 1])

            opHashNet.train()

            data_output = opHashNet(data)  # shape [n_points, n_output]
            if iter1 == 1 or iter1 % 14000 == 0:
                d_file = "{}/{}_dataset_{}.txt".format(net_folder, data_name, d_i)
                d_save = data_output.data.cpu().numpy()[0:1000, ]
                write_float(d_save, d_file)
                d_i += 1

            gt_rank = gt_rank1  # shape: 1*k2
            gt_rank_i = gt_rank - 1

            gt_rank = torch.from_numpy(gt_rank).float().to(device=device)
            relevance = torch.from_numpy(relevance).float()
            optimizer.zero_grad()

            loss_batch1, loss_batch2, loss_real1, loss_real2 = 0, 0, 0, 0
            for i in range(batch_size):
                # tt1 = time.time()
                gt2 = gt[i][gt_rank_i]
                data_kdb = data[gt2]  # the topk points need to be trained
                gt_relevance = relevance[gt_rank_i].to(device=device)

                add_query = queries[i].repeat(4, 1)
                query_output = opHashNet(add_query).reshape([4, n_output])
                query_output = query_output[0]
                # query_output = opHashNet(queries[i]).reshape([1, n_output])

                n_step = k2 // st
                ll = k2 - n_step
                ###########
                h2 = query_output - data_output
                h2 = torch.sum(h2.pow(2), dim=1, keepdim=True)  # shape [n_point, 1]
                ###########
                for j in range(0, k2, n_step):
                    h1 = query_output - data_output[gt2[j:j + n_step]]
                    h1 = torch.sum(h1.pow(2), dim=1).reshape([1, n_step])

                    hh = h1 - h2
                    ############# add new loss here ############
                    hh_new = sigmoid(h1 - h1.reshape([n_step, 1]))
                    matrix1 = torch.ones(n_step, n_step)
                    for ii in range(n_step):
                        for jj in range(ii + 1):
                            matrix1[jj, ii] = 0
                    matrix1 = matrix1.to('cuda')

                    loss2 = (hh_new * matrix1).mean()
                    #########################

                    s_hh = sigmoid(b * hh)
                    s_hh_a2 = ((s_hh.data >= 0.5) + 0).sum(axis=0)
                    r = s_hh.sum(dim=0) + 1  # shape[n_step]
                    #### save r #######
                    if (iter1 == 1 or iter1 % 14000 == 0) and i == 0:
                        h_file = "{}/{}_h_{}.txt".format(net_folder, data_name, r_i)
                        h_save = hh.data.cpu().numpy()[0:1000, ]
                        write_float(h_save, h_file)

                        r_file = "{}/{}_r_{}.txt".format(net_folder, data_name, r_i)
                        r_save = r.data.cpu().numpy().reshape([int(n_step / 10), 10])
                        write_float(r_save, r_file)
                        r_i += 1

                    gt_rank_sub = gt_rank[j:j + n_step]
                    loss1 = (((r - gt_rank_sub) * 0.2 + torch.log(1 + torch.exp((-2) * (r - gt_rank_sub) * 0.2)) - torch.log(torch.tensor(2.0))) * gt_relevance[j:j + n_step]).sum() / k2
                    
                    loss_real1 += torch.abs(s_hh_a2 + 1 - gt_rank_sub).mean()
                    loss1 = loss1 / batch_size
                    loss2 = (loss2 * alpha) / batch_size
                    loss = loss1 + loss2
                    loss_batch1 += loss1.data.cpu().item()
                    loss_batch2 += loss2.data.cpu().item()
                    if i == batch_size - 1 and j == ll:
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)

            optimizer.step()
            lr_print = optimizer.state_dict()["param_groups"][0]["lr"]
            time3 = time.time()
            print("|ep: %d" % epoch_i + "| step %d" % step_i + "| iter %d" % iter1 + "| b %d" % b +
                  " |lo1: %0.2f" % loss_batch1 +
                  " |lo2: %0.6f" % loss_batch2 +
                  " |lor1: %0.2f" % (loss_real1 / batch_size) +
                  " |rec: %0.4f" % recall +
                  " |b_rec: %0.4f" % best_recall +
                  " |lr: %0.6f" % lr_print +
                  " |t1: %0.2f" % (time2 - time1) +
                  " |t2: %0.2f" % (time3 - time2))

            writer.add_scalar(log_loss, loss_batch1, iter1)
            writer.add_scalar(log_loss2, loss_batch2, iter1)
            writer.add_scalar(log_real1, (loss_real1 / batch_size), iter1)
            writer.add_scalar(log_lr, lr_print, iter1)

        for name, param in opHashNet.named_parameters():
            writer.add_histogram(name + "_grad", param.grad.data.cpu().numpy(), epoch_i)
            writer.add_histogram(name + "_data", param.data.cpu().numpy(), epoch_i)

    return opHashNet


def build_index(data, idxs):
    n_points = data.shape[0]
    n_dim = data.shape[1]

    sorted_lists = []
    for d in range(n_dim):
        sorted_list = []
        for n in range(n_points):
            sorted_list.append([idxs[n], data[n][d]])
        sorted_list = sorted(sorted_list, key=lambda k: k[1])
        sorted_lists.append(sorted_list)
    return sorted_lists


def write_index(sorted_lists, index_id_file, index_value_file):
    index_ids, index_values = [], []
    for list in sorted_lists:
        index_ids.append(np.array(list)[:, 0].astype(int))
        index_values.append(np.array(list)[:, 1].astype(float))
    index_ids = np.array(index_ids)
    index_values = np.array(index_values)

    np.save(index_id_file, index_ids)
    np.save(index_value_file, index_values)


def load_index(index_id_file, index_value_file):
    index_ids = np.load(index_id_file)
    index_values = np.load(index_value_file)
    assert index_ids.shape[0] == index_values.shape[0]
    sorted_lists = []
    for ids, values in zip(index_ids, index_values):
        sorted_list = []
        for id, value in zip(ids, values):
            sorted_list.append([id, value])
        sorted_lists.append(sorted_list)
    return sorted_lists


def binary_search(sorted_list, value):
    low = 0
    high = len(sorted_list) - 1
    while low <= high:
        mid = int((low + high) / 2)
        if sorted_list[mid][1] == value:
            return mid, 0
        elif value < sorted_list[mid][1]:
            high = mid - 1
        else:
            low = mid + 1
    return mid, abs(sorted_list[mid][1] - value)


def get_distance(vector, vector_or_vectors):
    return ((vector - vector_or_vectors) ** 2).sum(-1)


def search(n_points, data_embeddings, queries, center_vectors, sorted_lists_all,
           query_embeds, knn, rerank_t, verify_t):
    best_ids, best_ids_dists = [], []
    # binary_time, scan_time, verify_time = 0, 0, 0
    n_iterate = 0
    for i, query_embed in enumerate(query_embeds):

        # t1 = time.time()
        # select the nearest clusters
        q_c_dists = ((queries[i] - center_vectors) ** 2).sum(1)  # k x 1
        sorted_lists = sorted_lists_all[np.argmin(q_c_dists)]

        n_lists = len(sorted_lists)  # m x 1
        priority_queue = []
        candidate_set = []
        Hits = np.zeros(n_points)
        Ranks = np.zeros(n_points)
        left_idx = [0] * n_lists
        right_idx = [0] * n_lists
        previous_ranks = [1] * n_lists
        for m, sorted_list in enumerate(sorted_lists):
            mid, dist = binary_search(sorted_list, query_embed[m])
            left_idx[m] = mid - 1
            right_idx[m] = mid + 1
            heappush(priority_queue, (dist, sorted_list[mid][0], 1, m))
        # t2 = time.time()
        while len(candidate_set) < rerank_t:
            n_iterate += 1
            dist, vertext_id, rank, list_id = heappop(priority_queue)
            Hits[vertext_id] += 1
            Ranks[vertext_id] += rank
            if Hits[vertext_id] >= n_lists:
                candidate_set.append([vertext_id, Ranks[vertext_id]])
                if len(candidate_set) >= rerank_t:
                    break
            current_list = sorted_lists[list_id]
            if left_idx[list_id] < 0 and right_idx[list_id] >= len(current_list):
                continue

            if left_idx[list_id] < 0:
                dist1 = 999999
            else:
                left_point = current_list[left_idx[list_id]]
                dist1 = abs(query_embed[list_id] - left_point[1])

            if right_idx[list_id] >= len(current_list):
                dist2 = 999999
            else:
                right_point = current_list[right_idx[list_id]]
                dist2 = abs(right_point[1] - query_embed[list_id])

            if dist1 < dist2:
                heappush(priority_queue, (dist1, left_point[0], previous_ranks[list_id] + 1, list_id))
                left_idx[list_id] -= 1
            else:
                heappush(priority_queue, (dist2, right_point[0], previous_ranks[list_id] + 1, list_id))
                right_idx[list_id] += 1
            previous_ranks[list_id] += 1
        # t3 = time.time()
        # reranking
        # candidate_set = sorted(candidate_set, key=lambda k:k[1])

        verify_set = nsmallest(verify_t, candidate_set, key=lambda k: k[1])
        for i, cand in enumerate(verify_set):
            verify_set[i][1] = get_distance_single(query_embed, data_embeddings[cand[0]])
        final_set = nsmallest(knn, verify_set, key=lambda k: k[1])
        final_ids = [cand[0] for cand in final_set]
        best_ids.append(final_ids)

    return best_ids


def get_recall(idxs, gt_ids):
    assert len(idxs) == len(gt_ids)
    r = 0
    for best_ids, gt in zip(idxs, gt_ids):
        count = 0
        assert len(best_ids) == len(gt)
        for gt_id in gt:
            if gt_id in best_ids:
                count += 1
        r += count / len(best_ids)
    recall = r / len(gt_ids)
    return recall


if __name__ == "__main__":
    data_name = "deep1b"
    output_folder = "/root/hdd-data/deep1b/Index_Learning_code"

    deep1b_base_path = "/root/hdd-data/deep1b/learn/base_data_1M.txt"
    deep1b_query_path = "/root/hdd-data/deep1b/learn/query_data_100.txt"
    deep1b_groundtruth_path = "/root/hdd-data/deep1b/learn/groundtruth_id_100.txt"

    deep1b_train_data = "/root/hdd-data/deep1b/learn/train_data_1.5M.txt"
    deep1b_train_query = "/root/hdd-data/deep1b/learn/train_query_700K.txt"

    # parameters:
    n_hidden = 512
    n_output = 25
    lr = 0.0001  # different dataset would need different learning rate
    epochs = 20
    batch_size = 50
    P = 1500000
    nq = 700000
    topK = 100
    S_ = 0

    # st is integer and >= 1, and should be tuned to make the memory used by pytorch is smaller than the actual machine memory
    # larger st would cost smaller memory, but training is slower.
    st = 1
    batch_normalization = False
    v = 10
    v_v = 38
    k = 100
    rerank_t = 1000
    verify_t = 300
    n_kmeans = 10
    n_kmeans_selected = 1
    alpha = 10

    log_file = "{}/log_{}_d{}_v{}_{}".format(output_folder, data_name, n_output, v, v_v)
    log_loss = "{}_loss_v{}_{}".format(data_name, v, v_v)
    log_loss2 = "{}_loss2_v{}_{}".format(data_name, v, v_v)
    log_real1 = "{}_loss1_v{}_{}".format(data_name, v, v_v)
    log_real2 = "{}_loss2_v{}_{}".format(data_name, v, v_v)
    log_recall = "{}_recall_v{}_{}".format(data_name, v, v_v)
    log_lr = "{}_lr_v{}_{}".format(data_name, v, v_v)
    net_folder = "{}/models_{}_d{}_v{}_{}".format(output_folder, data_name, n_output, v, v_v)
    os.makedirs(net_folder, exist_ok=True)
    print("start to load data .....")

    time1 = time.time()

    train_data = np.loadtxt(deep1b_train_data)
    # random_indices = torch.randperm(train_data.shape[0])
    # train_data = train_data[random_indices[:100000]]
    train_query = np.loadtxt(deep1b_train_query)
    test_data = np.loadtxt(deep1b_base_path)
    test_query = np.loadtxt(deep1b_query_path)
    test_gt = np.loadtxt(deep1b_groundtruth_path)
    print(train_data.shape)
    print(train_query.shape)
    print(test_data.shape)
    print(test_query.shape)
    print(test_gt.shape)

    mu = np.mean(train_data)
    sigma = np.std(train_data)
    z_normalize = lambda v: (v - mu) / sigma
    train_data, train_query = map(z_normalize, [train_data, train_query])

    mu2 = np.mean(test_data)
    sigma2 = np.std(test_data)
    z_normalize2 = lambda v: (v - mu2) / sigma2
    test_data, test_query = map(z_normalize2, [test_data, test_query])

    time2 = time.time()

    print("finish loading data with time %0.2f" % (time2 - time1))

    kmeans_file = "/root/hdd-data/deep1b/learn/base_kmeans.txt"
    kmeans = np.loadtxt(kmeans_file).astype(int)
    kmeans_partitions = [[] for _ in range(n_kmeans)]

    for i in range(kmeans.shape[0]):
        kmeans_partitions[kmeans[i][1]].append(kmeans[i][0])
    center_vectors = []

    # compute the center vectors
    for i in range(n_kmeans):
        center_vector = test_data[kmeans_partitions[i]].sum(0) / len(kmeans_partitions[i])
        assert len(center_vector) == test_data.shape[1]
        center_vectors.append(center_vector)
    center_vectors = np.array(center_vectors)  # k x d
    print("finished loading kmeans file and constructing centers....")

    net = neopfa(train_data, train_query, test_data, test_query, kmeans_partitions, center_vectors,
                 k, rerank_t, verify_t, n_kmeans, n_kmeans_selected, test_gt, n_hidden,
                 n_output, topK, S_, lr, epochs, batch_size, batch_normalization, log_file,
                 log_loss, log_loss2, log_real1, log_real2, log_recall, log_lr, net_folder, st, alpha)

    net_path = net_folder + "/" + data_name + "_last_iteration.NN"
    torch.save(net, net_path)

    print("finish training and saving the net ......")
