#include <cstring>
#include "NeOPFA_PLUS/PTA.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <cstdio>

using namespace std;

REAL *getInputData(const char *fileName, const int rowNum, const int colNum) {
    ifstream fileStream;
    REAL tmp = 0;
    int index = 0;
    float *X = new float[rowNum * colNum];
    fileStream.open(fileName, ios::in);
    if (fileStream.fail()) {
        cout << "file doesn't exist" << endl;
        fileStream.close();
        return 0;
    }
    while (index < rowNum * colNum) {
        fileStream >> tmp;
        X[index++] = tmp;
    }
    fileStream.close();
    return X;
}

int main(int args, char *argv[]) {
    char *data_name = argv[1];
    unsigned long int n_points = atoi(argv[2]);
    int n_dimensions = atoi(argv[3]);
    int M1 = atoi(argv[4]);
    int pagesize = atoi(argv[5]);

    char *index_folder = argv[6];
    char *data_file = argv[7];
    char *proj_data_file = argv[8];
    int K1 = atoi(argv[9]);
    char *kmeans_file = argv[10];

    cout << "data_name: " << data_name << endl;
    cout << "n_points: " << n_points << endl;
    cout << "n_dim: " << n_dimensions << endl;
    cout << "M1: " << M1 << endl;
    cout << "pagesize: " << pagesize << endl;
    cout << "index_folder: " << index_folder << endl;

    cout << "data_file: " << data_file << endl;
    cout << "proj_data_file: " << proj_data_file << endl;
    cout << "K1: " << K1 << endl;
    cout << "kmeans_file: " << kmeans_file << endl;
    cout << " the size of float: " << sizeof(float) << endl;

    if (access(index_folder, F_OK) != 0) {
        mkdir(index_folder, S_IRWXU);
    }
    timeval start, end;
    gettimeofday(&start, NULL);

    PTA *pta = new PTA(n_points, n_dimensions, M1, data_name, data_file, index_folder, pagesize, K1);
    pta->build_index(kmeans_file, proj_data_file);

    gettimeofday(&end, NULL);
    float index_time = diff_timeval(end, start);

    char index_time_file[256];
    sprintf(index_time_file, "%s/indexing_time.txt", index_folder);
    FILE *fp_time = fopen(index_time_file, "a+");
    FILE_CHEACK(fp_time, index_time_file);
    fprintf(fp_time, "the indexing time: %0.6f \n", index_time);
    fclose(fp_time);

    delete pta;
    printf("%s data indexing is done .........\n", data_name);
    printf("the indexing time is %f \n", index_time);

    return 0;

}
