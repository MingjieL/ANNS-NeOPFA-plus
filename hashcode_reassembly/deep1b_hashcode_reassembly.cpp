#include <iostream>
#include <fstream>
#include <stdio.h>
#include <random>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

using namespace std;

unsigned long int n_num = 1000000000;
// int n_dimensions = 128;
int m_dimensions = 25;
int K1 = 64;
int K2 = 256;

int main() {
    float *new_data = new float[n_num * m_dimensions];
    int *ori_map_new = new int[n_num];
    int *new_map_ori = new int[n_num];
    char *new_pro_file = "/root/hdd-data/deep1b/pro_kmeans_data/deep1b_new_base_d25_v10_vv38_9.bin";
    char *K1_num_file = "/root/hdd-data/deep1b/pro_kmeans_data/kmeans_num_K1.txt";
    char *K2_num_file = "/root/hdd-data/deep1b/pro_kmeans_data/kmeans_num_K2.txt";
    char *ori_map_new_file = "/root/hdd-data/deep1b/pro_kmeans_data/ori_map_new.ds";
    char *new_map_ori_file = "/root/hdd-data/deep1b/pro_kmeans_data/new_map_ori.ds";

    int K1_num[K1];  
    int K1_ptr_idx[K1];  
    memset(K1_ptr_idx, 0, sizeof(K1_ptr_idx));
    ifstream ifs1(K1_num_file, ios::in);
    for (int i = 0; i < K1; i++) {
        ifs1 >> K1_num[i];
        if (i > 0) {
            K1_ptr_idx[i] = K1_ptr_idx[i - 1] + K1_num[i - 1];
        }
    }
    if (K1_ptr_idx[K1 - 1] + K1_num[K1 - 1] != n_num) {
        cout << "K1 nums error" << endl;
        exit(0);
    }
    ifs1.close();

    int K2_num[K1][K2];  
    int K2_ptr_idx[K1][K2];  
    memset(K2_ptr_idx, 0, sizeof(K2_ptr_idx)); 
    ifstream ifs2(K2_num_file, ios::in);
    for (int i = 0; i < K1; i++) {
        for (int j = 0; j < K2; j++) {
            ifs2 >> K2_num[i][j];
            if (j > 0) {
                K2_ptr_idx[i][j] = K2_ptr_idx[i][j - 1] + K2_num[i][j - 1];
            }
        }
        if (K2_ptr_idx[i][K2 - 1] + K2_num[i][K2 - 1] != K1_num[i]) {
            cout << "K2 nums error" << endl;
            exit(0);
        }
    }
    ifs2.close();

    for (int k = 0; k < K1; k++) {
        char base_file[256];
        char map_file[256];
        char kmeans_file[256];
        sprintf(base_file, "/root/hdd-data/deep1b/pro_kmeans_data/data_%d", k);
        sprintf(map_file, "/root/hdd-data/deep1b/pro_kmeans_data/map_%d.txt", k);
        sprintf(kmeans_file, "/root/hdd-data/deep1b/pro_kmeans_data/kmeans_%d.txt", k);

        ifstream ifs_1(kmeans_file, ios::in);
        ifstream ifs_2(map_file, ios::in);
        FILE *file1 = fopen(base_file, "rb");
        float data[m_dimensions];
        int *K2_idx = new int[K1_num[k]];
        int *data_map = new int[K1_num[k]];

        for (int i = 0; i < K1_num[k]; i++) {
            ifs_1 >> K2_idx[i];  
            ifs_2 >> data_map[i];
            fread(data, sizeof(float), m_dimensions, file1);
            int obj_id = K1_ptr_idx[k] + K2_ptr_idx[k][K2_idx[i]];
            unsigned long int src_offsets = (unsigned long int)obj_id * m_dimensions;
            ori_map_new[data_map[i]] = obj_id;
            new_map_ori[obj_id] = data_map[i];
            // memcpy(new_data + src_offsets, data, sizeof(data));
            memcpy(new_data + src_offsets, data, 100);
            K2_ptr_idx[k][K2_idx[i]]++;
        }

        fclose(file1);
        ifs_1.close();
        ifs_2.close();
        delete[] K2_idx;
        delete[] data_map;
    }

    FILE *fp = fopen(new_pro_file, "wb");
    for (int i = 0; i < n_num; i++) {
        fwrite(new_data + (unsigned long int)i * m_dimensions, sizeof(float), m_dimensions, fp);
    }
    fclose(fp);

    FILE *fp1 = fopen(ori_map_new_file, "wb");
    fwrite(ori_map_new, sizeof(int), n_num, fp1);
    fclose(fp1);

    FILE *fp2 = fopen(new_map_ori_file, "wb");
    fwrite(new_map_ori, sizeof(int), n_num, fp2);
    fclose(fp2);

    // ofstream ofs1(ori_map_new_file, ios::out);
    // ofstream ofs2(new_map_ori_file, ios::out);
    // for (int i = 0; i < n_num; i++) {
    //     ofs1 << ori_map_new[i] << endl;
    //     ofs2 << new_map_ori[i] << endl;
    // }
    //
    // ofs1.close();
    // ofs2.close();

    delete[] new_data;
    delete[] ori_map_new;
    delete[] new_map_ori;
    return 0;
}
