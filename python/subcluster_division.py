from sklearn.cluster import KMeans
import numpy as np
import time
from collections import Counter
import random

K1 = 64
K2 = 256
dim = 25
SAMPLE_RATIO = 0.1
FIXED_SAMPLE_SIZE = 1000000


def kmeans(base_path, kmeans_file, cluster_stats_file):
    full_data = np.fromfile(base_path, dtype=np.float32).reshape(-1, dim)
    print(f"Origin data shape: {full_data.shape}")

    sample_size = min(FIXED_SAMPLE_SIZE, len(full_data))
    sample_indices = random.sample(range(len(full_data)), sample_size)
    sample_data = full_data[sample_indices]

    print(f"Sample data shape: {sample_size}")
    kmeans = KMeans(n_clusters=K2, random_state=42)
    kmeans.fit(sample_data)

    print("Start assigning cluster labels for all data.")
    all_labels = kmeans.predict(full_data)

    with open(kmeans_file, 'w') as output_file:
        for label in all_labels:
            output_file.write(f"{label}\n")

    counts = np.bincount(all_labels, minlength=K2)
    with open(cluster_stats_file, 'a') as f:
        f.write(' '.join(map(str, counts)) + '\n')


stats_file = "/root/hdd-data/sift1b/pro_kmeans_data/kmeans_num_K2.txt"
open(stats_file, 'w').close()

for i in range(K1):
    base_path = f"/root/hdd-data/sift1b/pro_kmeans_data/data_{i}"
    kmeans_file = f"/root/hdd-data/sift1b/pro_kmeans_data/kmeans_{i}.txt"

    start_time = time.time()
    kmeans(base_path, kmeans_file, stats_file)
    end_time = time.time()

    print(f"File {i} processed, time taken: {end_time - start_time:.4f} seconds")
    