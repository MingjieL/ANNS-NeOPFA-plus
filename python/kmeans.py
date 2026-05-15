from sklearn.cluster import KMeans
import numpy as np


n_kmeans = 10


def kmeans(base_path, kmeans_file):
    train_data = np.loadtxt(base_path)
    print(train_data.shape)

    kmeans = KMeans(n_clusters=n_kmeans, random_state=42)
    clusters = kmeans.fit_predict(train_data)

    with open(kmeans_file, 'w') as output_file:
        for i, cluster_id in enumerate(clusters):
            output_file.write(f"{i} {cluster_id}\n")

    print("KMeans clustering result has been saved to base_kmeans.txt")


sift1b_base_path = '/root/hdd-data/sift1b/learn/base_data_1M.txt'  # (1000000, 128)
sift1b_kmeans_file = "/root/hdd-data/sift1b/learn/base_kmeans.txt"

deep1b_base_path = '/root/hdd-data/deep1b/learn/base_data_1M.txt'  # (1000000, 96)
deep1b_kmeans_file = "/root/hdd-data/deep1b/learn/base_kmeans.txt"

Text2image_base_path = '/root/hdd-data/Text2image/learn/base_data_1M.txt'  # (1000000, 200)
Text2image_kmeans_file = "/root/hdd-data/Text2image/learn/base_kmeans.txt"

kmeans(sift1b_base_path, sift1b_kmeans_file)
kmeans(deep1b_base_path, deep1b_kmeans_file)
kmeans(Text2image_base_path, Text2image_kmeans_file)
