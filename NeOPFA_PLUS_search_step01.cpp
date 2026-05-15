#include <cstring>
#include "NeOPFA_PLUS/PTA.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <cstdio>
#include <filesystem>

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

bool createFileWithDirectory(const std::string& file_path) {
    try {
        std::filesystem::path path(file_path);
        std::filesystem::path dir = path.parent_path();
        
        if (!dir.empty() && !std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }
        
        std::ofstream file(file_path);
        if (file.is_open()) {
            file.close();
            return true;
        } else {
            std::cerr << "Can not create file: " << file_path << std::endl;
            return false;
        }
        
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "File system error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
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
    char *ori_map_new_file = argv[24];
    char *mostGroundtruthFile = argv[25];
    char *hamming_step02_id_file = argv[26];
    char *hamming_step02_file = argv[27];
    int useOriginalSpace = atoi(argv[28]); // 1 means using original space, 0 means using embedding space.
    int sT = atoi(argv[29]);
    int n_page = atoi(argv[30]);

    int rT = rr * vT; // 1 * 1200 = 1200

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

    createFileWithDirectory(hamming_step02_id_file);
    createFileWithDirectory(hamming_step02_file);

    printf("finished loading data....\n");
    PTA *pta = new PTA(n_points, n_dimensions, M1, m1, data_name, data_file, index_folder,
                       pagesize, nps, ratio_p, K1, n_selected, rT, vT, sT, sflag,
                       projDataFile, useOriginalSpace);

    printf("finished initialization....\n");

    ifstream ifs;
    ifs.open(mostGroundtruthFile, ios::in);
    if (ifs.fail()) {
        cout << "groundtruth file doesn't exist" << endl;
        ifs.close();
        exit(0);
    }
    int most_groundtruth_id[n_queries];
    for (int i = 0; i < n_queries; i++) {
        ifs >> most_groundtruth_id[i];
    }
    ifs.close();

    FILE *map_fp = fopen(ori_map_new_file, "rb");
    FILE_CHEACK(map_fp, ori_map_new_file);
    int *ori_map_new = new int[n_points];
    fread(ori_map_new, sizeof(int), n_points, map_fp);

    fclose(map_fp);

    BEGIN_MEM_MONITOR();
    int **test01 = new int *[n_queries];
    float hit_rate = 0.0f;
    float search_time = 0.0;
    pta->preprocessing();
    for (int q = 0; q < n_queries; ++q) {
        pta->initialize();
        float time_i;
        test01[q] = new int[sT];
        pta->knn_search_step01(&proj_query[q * M1], n_IO[q], most_groundtruth_id[q], ori_map_new, hamming_step02_id_file,
                               hamming_step02_file, hit_rate, test01[q], time_i, n_page);
        search_time += time_i;
        cout << "finished query " << q << " with time " << time_i << endl;
        pta->release();
    }
    pta->postprocessing();
    END_MEM_MONITOR();

    printf("query searching is done .........\n");

    search_time /= n_queries;

    int n_IO_total = 0;

    for (int q = 0; q < n_queries; ++q) {
        n_IO_total += n_IO[q];
    }

    // compute the recall
    int *n_correct_test01 = new int[n_queries];
    memset(n_correct_test01, 0, sizeof(int) * n_queries);

    for (int i1 = 0; i1 < n_queries; i1++) {
        for (int j1 = 0; j1 < knn; j1++) {
            for (int k1 = 0; k1 < sT; k1++) {
                if (gt_id[i1 * knn + j1] == test01[i1][k1]) {
                    n_correct_test01[i1]++;
                    break;
                }
            }
        }
    }

    int n_correct_test01_sum = 0;
    for (int i = 0; i < n_queries; ++i) {
        n_correct_test01_sum += n_correct_test01[i];
    }

    float recall_test01 = n_correct_test01_sum * 1.0f / (knn * n_queries);

    float hit_rat_avg = hit_rate / n_queries;

    printf("The performance of PTANNS is bellow: \n");

    int IO_final = n_IO_total / n_queries;

    printf("the knn is %d. \n", knn);
    printf("the recall test01 is %f. \n", recall_test01);
    printf("the IO cost is %d. \n", IO_final);
    printf("the query time is %f. \n", search_time);
    printf("the number of evaluation is %d. \n", vT);
    printf("the avg of hit rate is %f. \n", hit_rat_avg);
    printf("the peak memory usage is %ld MB. \n", (g_peak_rss_pages * page)/1024/1024);

    FILE *fp2;
    fp2 = fopen(out_file, "a+");
    fprintf(fp2, "%s: %d %0.6f %d %0.6f #N_%d %0.6f %ldMB\n",
            data_name, knn, recall_test01, IO_final, search_time, vT, hit_rat_avg, (g_peak_rss_pages * page)/1024/1024);
    fclose(fp2);

    printf("finished releasing \n");

    for (int i1 = 0; i1 < n_queries; i1++) {
        delete[] test01[i1];
    }
    delete[] test01;
    delete[] ori_map_new;
    delete[] n_correct_test01;
    delete[] n_IO;
    delete pta;
    delete[] queries;
    delete[] data_t;
    delete[] gt_id;
    delete[] gt_dist;

    printf("releasing is done ...........\n");

    return 0;
}
