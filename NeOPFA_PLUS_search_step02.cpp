#include <cstring>
#include "NeOPFA_PLUS/PTA.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <cstdio>

using namespace std;

static std::atomic<bool> g_monitor_mem(false);
static std::atomic<size_t> g_peak_vm_pages(0);
static std::atomic<size_t> g_peak_rss_pages(0);
static std::thread g_mem_thread;

void mem_monitor() {
    g_peak_vm_pages = 0;
    g_peak_rss_pages = 0;
    while (g_monitor_mem) {
        FILE* fp = fopen("/proc/self/statm", "r");
        if (!fp) {
            usleep(10000);
            continue;
        }
        size_t vm, rss;
        if (fscanf(fp, "%zu %zu", &vm, &rss) == 2) {
            if (vm > g_peak_vm_pages) g_peak_vm_pages = vm;
            if (rss > g_peak_rss_pages) g_peak_rss_pages = rss;
        }
        fclose(fp);
        usleep(10000);
    }
}

#define BEGIN_MEM_MONITOR() \
g_monitor_mem = true; \
g_mem_thread = std::thread(mem_monitor);

#define END_MEM_MONITOR() \
g_monitor_mem = false; \
if (g_mem_thread.joinable()) g_mem_thread.join(); \
size_t page = sysconf(_SC_PAGESIZE); \
// std::cout << "the peak RSS: " << (g_peak_rss_pages * page)/1024/1024 << "MB\n";

float *getInputData_f(char *fileName, int rowNum, int colNum, int K) {
    ifstream fileStream;
    float tmp = 0;
    LONGINT index = 0;
    float *X_t = new float[rowNum * K];
    float *X = new float[rowNum * colNum];
    fileStream.open(fileName, ios::in);
    if (fileStream.fail()) {
        cout << "file doesn't exist" << endl;
        fileStream.close();
        return 0;
    }
    while (index < rowNum * K) {
        fileStream >> tmp;
        X_t[index++] = tmp;
    }
    fileStream.close();

    for (int i = 0; i < rowNum; ++i) {
        for (int j = 0; j < colNum; ++j) {
            X[i * colNum + j] = X_t[i * K + j];
        }
    }
    delete[]X_t;

    return X;
}

int *getInputData_i(char *fileName, int rowNum, int colNum, int K) {
    ifstream fileStream;
    int tmp = 0;
    LONGINT index = 0;
    int *X_t = new int[rowNum * K];
    int *X = new int[rowNum * colNum];
    fileStream.open(fileName, ios::in);
    if (fileStream.fail()) {
        cout << "file doesn't exist" << endl;
        fileStream.close();
        return 0;
    }
    while (index < rowNum * K) {
        fileStream >> tmp;
        X_t[index++] = tmp;
    }
    fileStream.close();

    for (int i = 0; i < rowNum; ++i) {
        for (int j = 0; j < colNum; ++j) {
            X[i * colNum + j] = X_t[i * K + j];
        }
    }
    delete[]X_t;

    return X;
}

int main(int args, char *argv[]) {
    char *data_name = argv[1];
    unsigned long int n_points = atoi(argv[2]);
    int n_queries = atoi(argv[3]);
    int n_dimensions = atoi(argv[4]);

    int M1 = atoi(argv[5]);
    int m1 = atoi(argv[6]);
    int knn = atoi(argv[7]);
    int vT = atoi(argv[8]);

    int pagesize = atoi(argv[9]);
    int nps = atoi(argv[10]);
    int ratio_p = atoi(argv[11]);

    char *data_file = argv[12];
    char *proj_query_file = argv[13];
    char *query_file = argv[14];
    char *index_folder = argv[15];

    char *gt_id_file = argv[16];
    char *gt_dist_file = argv[17];
    char *out_file = argv[18];
    int rr = atoi(argv[19]);

    int K1 = atoi(argv[20]);
    int n_selected = atoi(argv[21]);
    int sflag = atoi(argv[22]); // whether consider the selected centroids equally
    char *projDataFile = argv[23];
    char *hamming_step02_id_file = argv[24];
    char *hamming_step02_lable_file = argv[25];
    int alpha = atoi(argv[26]);
    int useOriginalSpace = atoi(argv[27]); // 1 means using original space, 0 means using embedding space.
    int sT = atoi(argv[28]);
    int rT = rr * vT; // 1 * 1200 = 1200

    Result **knn_results = new Result *[n_queries];
    for (int k = 0; k < n_queries; ++k) {
        knn_results[k] = new Result[knn];
    }

    int *n_IO = new int[n_queries];
    REAL *queries = new REAL[n_queries * n_dimensions];
    float *data_t = new float[n_dimensions];

    memset(n_IO, 0, sizeof(int) * n_queries);

    FILE *query_fp = fopen(query_file, "rb");
    FILE_CHEACK(query_fp, query_file);
    fread(queries, sizeof(REAL), n_queries * n_dimensions, query_fp);
    fclose(query_fp);

    REAL *proj_query = getInputData_f(proj_query_file, n_queries, M1, M1);
    float *gt_dist = getInputData_f(gt_dist_file, n_queries, knn, 100);
    int *gt_id = getInputData_i(gt_id_file, n_queries, knn, 100);

    printf("finished loading data....\n");
    PTA *pta = new PTA(n_points, n_dimensions, M1, m1, data_name, data_file, index_folder,
                       pagesize, nps, ratio_p, K1, n_selected, rT, vT, sT, sflag,
                       projDataFile, useOriginalSpace);
    // pta->initialize();
    printf("finished initialization....\n");

    ifstream infile1, infile2;
    infile1.open(hamming_step02_id_file, ios::in);
    infile2.open(hamming_step02_lable_file, ios::in);
    if (!infile1.is_open() || !infile2.is_open()) {
        cout << "file open fail" << endl;
        exit(0);
    }

    int data_step02_id[n_queries][sT];
    float **data_step02_label = new float *[n_queries * sT];
    for (int i = 0; i < n_queries; i++) {
        for (int j = 0; j < sT; j++) {
            infile1 >> data_step02_id[i][j];
        }
    }
    for (int i = 0; i < n_queries * sT; i++) {
        data_step02_label[i] = new float[n_dimensions];
        for (int j = 0; j < n_dimensions; j++) {
            infile2 >> data_step02_label[i][j];
        }
    }

    infile1.close();
    infile2.close();

    float verify_time = 0;
    timeval start, end, start_i, end_i;

    BEGIN_MEM_MONITOR();
    gettimeofday(&start, NULL);
    for (int q = 0; q < n_queries; ++q) {
        gettimeofday(&start_i, NULL);
        pta->knn_search_step02(&proj_query[q * M1], &queries[q * n_dimensions], knn, knn_results[q], n_IO[q],
                               verify_time, data_step02_id[q], alpha, data_step02_label, q, sT);
        gettimeofday(&end_i, NULL);
        float time_i = diff_timeval(end_i, start_i);
        cout << "finished query " << q << " with time " << time_i << endl;
    }
    gettimeofday(&end, NULL);
    END_MEM_MONITOR();

    float search_time = diff_timeval(end, start);
    printf("query searching is done .........\n");

    search_time /= n_queries;
    verify_time /= n_queries;

    int n_IO_total = 0;
    for (int q = 0; q < n_queries; ++q) {
        n_IO_total += n_IO[q];
    }
    printf("The performance of PTANNS is bellow: \n");

    // compute the ratio
    // compute the original distance first for the case of useOriginal==0
    FILE *data_fp;
    if (!useOriginalSpace) {
        data_fp = fopen(data_file, "rb");
    }
    int i, j, k;
    float dist1, dist2;
    float ratio_average = 0.0;
    for (i = 0; i < n_queries; ++i) {
        float ratio_ = 0.0;
        if (!useOriginalSpace) {
            for (j = 0; j < knn; ++j) {
                unsigned long int ID = knn_results[i][j].id;
                unsigned long int size2 = sizeof(float) * ID * n_dimensions;
                fseek(data_fp, size2, SEEK_SET);
                fread(data_t, sizeof(float), n_dimensions, data_fp);
                knn_results[i][j].dist = compute_distance(&queries[i * n_dimensions], data_t, n_dimensions);
            }
        }
        sort(knn_results[i], knn_results[i] + knn, rcmp);
        for (j = 0; j < knn; ++j) {
            dist1 = sqrt(knn_results[i][j].dist);
            dist2 = sqrt(gt_dist[i * knn + j]);
            if (fabs(dist1 - ZERO) < FLOATZERO)
                ratio_ += 1;
            else if (fabs(dist2 - ZERO) < FLOATZERO)
                ratio_ += 1;
            else
                ratio_ += dist1 / dist2;
        }
        ratio_ /= knn;
        ratio_average += ratio_;
    }
    if (!useOriginalSpace) {
        fclose(data_fp);
    }
    ratio_average /= n_queries;

    // compute the recall
    int *n_correct = new int[n_queries];
    memset(n_correct, 0, sizeof(int) * n_queries);

    for (i = 0; i < n_queries; ++i) {
        for (j = 0; j < knn; ++j) {
            for (k = 0; k < knn; ++k) {
                if (gt_id[i * knn + j] == knn_results[i][k].id) {
                    n_correct[i]++;
                    break;
                }
            }
        }
    }

    int n_correct_sum = 0;
    for (i = 0; i < n_queries; ++i) {
        n_correct_sum += n_correct[i];
    }

    float recall = n_correct_sum * 1.0f / (knn * n_queries);
    int IO_final = n_IO_total / n_queries;
    float evaluation_ratio = vT * 1.0f / n_points;

    printf("the knn is %d. \n", knn);
    printf("the recall is %f. \n", recall);
    printf("the ratio is %f. \n", ratio_average);
    printf("the IO cost is %d. \n", IO_final);
    printf("the query time is %f. \n", search_time);
    printf("the verify time is %f. \n", verify_time);
    printf("the number of evaluation is %d. \n", vT);
    printf("the ratio of evaluation is %f. \n", evaluation_ratio);
    printf("the peak memory usage is %ld MB. \n", (g_peak_rss_pages * page)/1024/1024);

    FILE *fp2;
    fp2 = fopen(out_file, "a+");
    fprintf(fp2, "%s: %d %0.6f %0.6f %d %0.6f %0.6f #N_%d %0.6f %ldMB\n",
            data_name, knn, recall, ratio_average, IO_final, search_time, verify_time, vT, evaluation_ratio, (g_peak_rss_pages * page)/1024/1024);
    fclose(fp2);

    printf("finished releasing \n");

    for (int i1 = 0; i1 < n_queries * sT; i1++) {
        delete[] data_step02_label[i1];
    }
    delete[] data_step02_label;

    for (int l = 0; l < n_queries; ++l) {
        delete[] knn_results[l];
    }
    delete[] knn_results;
    delete[] n_IO;
    delete pta;
    delete[] queries;
    delete[] data_t;
    delete[] gt_id;
    delete[] gt_dist;
    delete[] n_correct;

    printf("releasing is done ...........\n");

    return 0;
}
