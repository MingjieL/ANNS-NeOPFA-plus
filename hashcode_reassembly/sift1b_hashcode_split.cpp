#include <iostream>
#include <fstream>
#include <stdio.h>
#include <random>
#include <ctime>

using namespace std;

int n_num = 1000000000;
// int n_dimensions = 128;
int m_dimensions = 25;
int K1 = 64;

void openFile(FILE** file, ofstream *ofs) {
    for (int i = 0; i < K1; i++) {
        char base_file[256];
        char map_file[256];
        sprintf(base_file, "/root/hdd-data/sift1b/pro_kmeans_data/data_%d", i);
        sprintf(map_file, "/root/hdd-data/sift1b/pro_kmeans_data/map_%d.txt", i);
        file[i] = fopen(base_file, "wb");
        ofs[i].open(map_file, ios::out);
    }
}

void closeFile(FILE** file, ofstream *ofs) {
    for (int i = 0; i < K1; i++) {
        fclose(file[i]);
        ofs[i].close();
    }
}

void addData(FILE** file, ofstream *ofs, float* data, int i, int kmeans_id) {
    fwrite(data, sizeof(float), m_dimensions, file[kmeans_id]);
    ofs[kmeans_id] << i << endl;
}

int main() {
    char *base_file = "/root/hdd_data/sift1b/data.ds";
    char *pro_base_file = "/root/hdd-data/sift1b/Index_Learning_code/models_sift1b_d25_v10_38/sift1b_base_d25_v10_vv38_5.bin";
    char *kmeans_file = "/root/hdd-data/sift1b/forest_64_256_8/codes1.txt";
    char *kmeans_num_file = "/root/hdd-data/sift1b/pro_kmeans_data/kmeans_num_K1.txt";

    int *kmeans_id = new int[n_num];

    FILE **file = new FILE *[K1];
    ofstream *ofs = new ofstream[K1];

    openFile(file, ofs);

    float *data = new float[m_dimensions];
    int kmeans_num[K1] = {0};

    FILE *file1 = fopen(pro_base_file, "rb");
    ifstream ifs1(kmeans_file, ios::in);
    int p_id;
    for (int i = 0; i < n_num; i++) {
        ifs1 >> p_id >> kmeans_id[i];
        if (kmeans_id[i] < 0 || kmeans_id[i] >= K1) {
            cout << "kmeans_id = " << kmeans_id[i] << endl;
            exit(0);
        }
        kmeans_num[kmeans_id[i]]++;
        if (i != 0 && i % 1000000 == 0) {
            cout << "finished " << i << " points" << endl;
        }
    }

    ofstream ofs1(kmeans_num_file, ios::out);
    for (int i = 0; i < K1; i++) {
        ofs1 << kmeans_num[i] << endl;
    }
    ofs1.close();

    for (int i = 0; i < n_num; i++) {
        fread(data, sizeof(float), m_dimensions, file1);
        addData(file, ofs, data, i, kmeans_id[i]);
        if (i != 0 && i % 1000000 == 0) {
            cout << "finished " << i << " points" << endl;
        }
    }

    fclose(file1);
    ifs1.close();
    closeFile(file, ofs);
    delete[] kmeans_id;
    delete[] file;
    delete[] ofs;
    delete[] data;
    return 0;
}
