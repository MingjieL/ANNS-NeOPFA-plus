#ifndef PTANNS_PTA_H
#define PTANNS_PTA_H

#include <algorithm>
#include <float.h>
#include <math.h>
#include <cstdio>
#include <stdio.h>
#include <cstring>
#include <sys/time.h>
#include <unistd.h>
#include <queue>
#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <pthread.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>

using namespace std;

#define REAL float
#define LONGINT int
//#define PAGESIZE 4096  // 4K
//#define NPS 100 // the number of pagesize for the cache
//#define RATIO 100 // the ratio for sequential IO
#define ZERO 0.0
#define FLOATZERO 1e-4
#define FILE_CHEACK(fp, message) {if (fp == NULL) {printf("Something wrong when reading file in %s\n", message); exit(1);}}
#define FLOATMAX 1e9

template<typename T>
struct res_pair_raw {
    int id;
    T dist;

    bool operator>(const res_pair_raw<T> &) const;

    bool operator>=(const res_pair_raw<T> &) const;

    bool operator==(const res_pair_raw<T> &) const;

    bool operator<=(const res_pair_raw<T> &) const;

    bool operator<(const res_pair_raw<T> &) const;
};

struct _pair {
    float dist;
    int id;

    friend bool operator<(_pair n1, _pair n2) {
        return n1.dist > n2.dist;
    }
};

struct BlockId {
    int cache1_i;
    int cache1_i_s;
    int cache2_i;
    int cache2_i_s;
    int list_left_i;
    int list_right_i;
    int para_left_i;
    int para_right_i;
};

struct KeyValue {
    int key;
    int value;

    bool operator<(const KeyValue& other) const {
        return value < other.value;
    }
};

class search_node {
public:
    int *ids;
    int num_ids;
    int list_m;
    double priority_value;
    int direction;

    bool operator>(const search_node &) const;

    bool operator>=(const search_node &) const;

    bool operator==(const search_node &) const;

    bool operator<=(const search_node &) const;

    bool operator<(const search_node &) const;

public:
    search_node();

    search_node(int *ids_, int num_, int list_m_, double priority_, int dir);

//    ~search_node();
    void release();
};

typedef struct Point {
    int id;
    float coordinate;
};

typedef struct PPoint {
    int id;
    double coordinate;
};

struct Result {
    LONGINT id;
    REAL dist;
};

struct newResult {
    LONGINT id;
    LONGINT num;
    REAL dist;
};

void check_file(char *_fn);

void check_folder(char *_fn);

bool rcmp(Result a, Result b);

float diff_timeval(timeval t1, timeval t2);

REAL compute_distance(float *x, float *y, int dimension);

float compute_distance(float *x, int dimension);


class PTA {
private:
    unsigned long int n_points;
    int n_dimensions;
    int M1;
    int m1;

    char *data_f;
    char *proj_data_f;
    char *index_f;
    char *data_name;

    int *cache1_i;
    int *cache1_i_s;
    int *cache2_i;
    int *cache2_i_s;
    int *list_left_i;
    int *list_right_i;
    int *para_left_i;
    int *para_right_i;

    char *C;
    float *O;
    BlockId *blockId;
    float *data;
    float *data_proj;
    float *data_proj_new;

    float **parameters;
    int tail_size;
    int number_of_blocks;
    // int *caches;
    int **caches;
    int **cache1;
    int **cache2;
    int cache_size;
    int cache_size_0;

    int *rerank_id;
    int *verify_id;

    FILE **index_fps_l;
    FILE **index_fps_r;

    FILE **fp_para;
    FILE ** *index_fps_all_l;
    FILE ** *index_fps_all_r;

    FILE ** *fp_para_all;
    FILE *data_fp;
    FILE *proj_data_fp;
    int proj_data_fd;
    struct stat st;
    float *mapped;

    int PAGESIZE;
    int NPS;
    int RATIO;
    int K1;
    int n_selected_cen;
    int rT;
    int vT;
    int sT;
    int sflag;
    int useOriginal;

    // add
    float **centroid;
    int *n_points_cen;
    unsigned long int offset;
    char centroid_file[256];
    char index_file_l[256];
    char index_file_r[256];
    char para_file[256];
    float **temp_data;
    newResult *knn_result;
    priority_queue<search_node> *PTA_queue;
    vector<vector<int>> clusters;
    vector<KeyValue> pair_map_data;

public:
    PTA(unsigned long int n_points,
        int n_dimensions,
        int m1,
        char *data_name,
        char *data_f,
        char *index_f,
        int pagesize, int K1_);

    PTA(unsigned long int n_points,
        int n_dimensions,
        int M1,
        int m1,
        char *data_name,
        char *data_f,
        char *index_f,
        int pagesize,
        int nps,
        int ratio_, int K1_, int n_selected_cen_,
        int rT, int vT, int sT, int sflag_,
        char *proj_data_f, int useOriginal);

    PTA();

    ~PTA();
    void preprocessing();

    void initialize();

    void release();

    void postprocessing();

    PPoint binary_search(float *array, REAL key);

    unsigned int binary_search2(float *array, REAL key);

    search_node linear_scan_from_cache(float *parameters, int &para_left_i, int &para_right_i,
                                       float key, int *cache1, int *cache2, int &cache1_i,
                                       int &cache1_i_s, int &cache2_i, int &cache2_i_s, int m_, int page);

    int linear_scan_from_file(FILE *fp, int &list_i, int *cache, int c_size, int &cache_i, int direction);

    void build_index(char *kmeans_file, char *proj_data_file);

    void knn_search_step01(float *proj_query, int &IO, int most_groundtruth_id, int *ori_map_new,
                            char *hamming_step02_id_file, char *hamming_step02_file, float &hit_rate,
                            int *test01, float &time_i, int n_page);

    void knn_search_step02(float *proj_query, REAL *query, int knn, Result *knn_result,
                          int &IO, float &vefiry_time, int *data_step02_id, int alpha,
                          float **data_step02_label, int q, int sT);
};

#endif