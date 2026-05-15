#include <queue>
#include "PTA.h"

template<typename T>
bool res_pair_raw<T>::operator>(const res_pair_raw<T> &n) const {
    return (dist > n.dist);
}

template<typename T>
bool res_pair_raw<T>::operator>=(const res_pair_raw<T> &n) const {
    return (dist >= n.dist);
}

template<typename T>
bool res_pair_raw<T>::operator==(const res_pair_raw<T> &n) const {
    return (dist == n.dist);
}

template<typename T>
bool res_pair_raw<T>::operator<=(const res_pair_raw<T> &n) const {
    return (dist <= n.dist);
}

template<typename T>
bool res_pair_raw<T>::operator<(const res_pair_raw<T> &n) const {
    return (dist < n.dist);
}

template<>
bool res_pair_raw<long long>::operator>(const res_pair_raw<long long> &n) const {
    return (dist > n.dist);
}

template<>
bool res_pair_raw<long long>::operator>=(const res_pair_raw<long long> &n) const {
    return (dist >= n.dist);
}

template<>
bool res_pair_raw<long long>::operator==(const res_pair_raw<long long> &n) const {
    return (dist == n.dist);
}

template<>
bool res_pair_raw<long long>::operator<=(const res_pair_raw<long long> &n) const {
    return (dist <= n.dist);
}

template<>
bool res_pair_raw<long long>::operator<(const res_pair_raw<long long> &n) const {
    return (dist < n.dist);
}

template<>
bool res_pair_raw<double>::operator>(const res_pair_raw<double> &n) const {
    return (dist > n.dist);
}

template<>
bool res_pair_raw<double>::operator>=(const res_pair_raw<double> &n) const {
    return (dist >= n.dist);
}

template<>
bool res_pair_raw<double>::operator==(const res_pair_raw<double> &n) const {
    return (dist == n.dist);
}

template<>
bool res_pair_raw<double>::operator<=(const res_pair_raw<double> &n) const {
    return (dist <= n.dist);
}

template<>
bool res_pair_raw<double>::operator<(const res_pair_raw<double> &n) const {
    return (dist < n.dist);
}

bool search_node::operator>(const search_node &n) const {
    return priority_value < n.priority_value;
}

bool search_node::operator>=(const search_node &n) const {
    return priority_value <= n.priority_value;
}

bool search_node::operator==(const search_node &n) const {
    return priority_value == n.priority_value;
}

bool search_node::operator<=(const search_node &n) const {
    return priority_value >= n.priority_value;
}

bool search_node::operator<(const search_node &n) const {
    return priority_value > n.priority_value;
}

search_node::search_node(int *ids_, int num_, int list_m_, double priority_, int dir) {
    // ids = new int[num_];
    // num_ids = num_;
    // memcpy(ids, ids_, sizeof(int) * num_);
    // list_m = list_m_;
    // priority_value = priority_;
    // direction = dir;

    ids = ids_;
    num_ids = num_;
    list_m = list_m_;
    priority_value = priority_;
    direction = dir;
}

search_node::search_node() {
    ids = NULL;
}

void search_node::release() {
    // delete[] ids;
}


bool qcmps(Point a, Point b) {
    return a.coordinate < b.coordinate;
}

bool qcmpr(Point a, Point b) {
    return a.coordinate > b.coordinate;
}

bool rcmp(Result a, Result b) {
    return a.dist < b.dist;
}

bool qcmp2(int a, int b) {
    return a < b;
}

float diff_timeval(timeval t1, timeval t2) {
    return (float) (t1.tv_sec - t2.tv_sec) + (t1.tv_usec - t2.tv_usec) * 1e-6;
}

void check_file(char *_fn) {
    if (access(_fn, F_OK) != 0) {
        printf("The file %s does not exist.\n", _fn);
        printf("Please check !\n");
        exit(-1);
    }
}

void check_folder(char *_fn) {
    if (access(_fn, F_OK) != 0) {
        printf("The folder %s does not exist.\n", _fn);
        printf("Please check !\n");
        exit(-1);
    }
}

PTA::PTA(unsigned long int n, int d, int M1, char *name,
         char *d_f, char *i_f, int pagesize, int K1_) {
    n_points = n;
    n_dimensions = d;
    this->M1 = M1;
    data_name = name;
    data_f = d_f;
    index_f = i_f;
    PAGESIZE = pagesize;
    K1 = K1_;
}

PTA::PTA(unsigned long int n, int d, int M1, int m1, char *name, char *d_f, char *i_f,
         int pagesize, int nps, int ratio_, int K1_, int n_selected_cen_,
         int rT_, int vT_, int sT_, int sflag_, char *proj_data_f_, int useOriginal_) {
    n_points = n;
    n_dimensions = d;
    this->M1 = M1;
    this->m1 = m1;
    data_name = name;
    data_f = d_f;
    index_f = i_f;
    PAGESIZE = pagesize;
    NPS = nps;
    RATIO = ratio_;
    K1 = K1_;
    n_selected_cen = n_selected_cen_;
    rT = rT_;
    vT = vT_;
    sT = sT_;
    sflag = sflag_;

    this->proj_data_f = proj_data_f_;
    this->useOriginal = useOriginal_;
}

PTA::PTA() {
}

PTA::~PTA() {
}

REAL compute_distance(REAL *x, REAL *y, int dimension) {
    float p = 0.0;
    for (int i = 0; i < dimension; ++i) {
        float temp = x[i] - y[i];
        p += temp * temp;
    }
    return p;
}

void PTA::preprocessing() {
    // allocate memory
    C = new char[n_points];

    index_fps_all_r = new FILE **[K1];
    index_fps_all_l = new FILE **[K1];
    fp_para_all = new FILE **[K1];

    for (int i = 0; i < K1; i++) {
        index_fps_all_r[i] = new FILE*[M1];
        index_fps_all_l[i] = new FILE*[M1];
        fp_para_all[i] = new FILE*[M1];
        for (int j = 0; j < M1; j++) {
            sprintf(index_file_r, "%s/%s_%d_%d.index", index_f, data_name, i, j);
            sprintf(index_file_l, "%s/%s_%d_%d.index", index_f, data_name, i, j);
            sprintf(para_file, "%s/%s_%d_%d.para", index_f, data_name, i, j);
            index_fps_all_r[i][j] = fopen(index_file_r, "rb");
            index_fps_all_l[i][j] = fopen(index_file_l, "rb");
            fp_para_all[i][j] = fopen(para_file, "rb");
            FILE_CHEACK(index_fps_all_r[i][j], index_file_r);
            FILE_CHEACK(index_fps_all_l[i][j], index_file_l);
            FILE_CHEACK(fp_para_all[i][j], para_file);
        }
    }

    proj_data_fp = fopen(proj_data_f, "rb");
}

void PTA::initialize() {
    index_fps_r = new FILE *[M1];
    index_fps_l = new FILE *[M1];
    fp_para = new FILE *[M1];

    memset(C, 0, sizeof(char) * n_points);
    // memset(O, 0, sizeof(float) * n_points);

    blockId = new BlockId[M1];
    cache1_i = new int[M1];
    cache1_i_s = new int[M1];
    cache2_i = new int[M1];
    cache2_i_s = new int[M1];
    list_left_i = new int[M1];
    list_right_i = new int[M1];
    para_left_i = new int[M1];
    para_right_i = new int[M1];

    cache_size = NPS * PAGESIZE / sizeof(int);

    data = new float[n_dimensions];
    data_proj = new float[M1];
    data_proj_new = new float[cache_size];


    caches = new int *[M1];
    for (int m = 0; m < M1; ++m) {
        caches[m] = new int[PAGESIZE / sizeof(int)];
    }

    rerank_id = new int[rT];
    verify_id = new int[vT];

    // add
    cache_size_0 = PAGESIZE / sizeof(int);
    centroid = new float *[K1];
    n_points_cen = new int[K1];
    temp_data = new float *[rT];
    knn_result = new newResult[sT];
    cache1 = new int *[M1];
    cache2 = new int *[M1];
    parameters = new float *[M1];
    for (int m = 0; m < M1; ++m) {
        cache1[m] = new int[cache_size];
        cache2[m] = new int[cache_size];
    }

    for (int i = 0; i < rT; i++) {
        temp_data[i] = new float[M1];
    }

    for (int i = 0; i < sT; ++i) {
        knn_result[i].id = -1;
        knn_result[i].dist = FLOATMAX;
    }

    for (int i = 0; i < K1; ++i) {
        centroid[i] = new float[n_dimensions];
        sprintf(centroid_file, "%s/%s_%d.centroid", index_f, data_name, i);
        FILE *cen_fp = fopen(centroid_file, "rb");
        FILE_CHEACK(cen_fp, centroid_file);
        fread(centroid[i], sizeof(float), n_dimensions, cen_fp);
        fread(&n_points_cen[i], sizeof(int), 1, cen_fp);
        fclose(cen_fp);
    }

    memset(blockId, 0, sizeof(BlockId) * M1);
    memset(cache1_i, 0, sizeof(int) * M1);
    memset(cache1_i_s, 0, sizeof(int) * M1);
    memset(cache2_i, 0, sizeof(int) * M1);
    memset(cache2_i_s, 0, sizeof(int) * M1);
    memset(list_left_i, 0, sizeof(int) * M1);
    memset(list_right_i, 0, sizeof(int) * M1);
    memset(para_left_i, 0, sizeof(int) * M1);
    memset(para_right_i, 0, sizeof(int) * M1);

    PTA_queue = new priority_queue<search_node>;
    pair_map_data.reserve(rT);
}

void PTA::release() {
    // release
    delete[] blockId;
    delete[] data;
    delete[] data_proj;
    delete[] data_proj_new;

    delete[] cache1_i;
    delete[] cache1_i_s;
    delete[] cache2_i;
    delete[] cache2_i_s;
    delete[] list_left_i;
    delete[] list_right_i;
    delete[] para_left_i;
    delete[] para_right_i;

    for (int m = 0; m < M1; m++) {
        delete[] cache1[m];
        delete[] cache2[m];
    }
    delete[] cache1;
    delete[] cache2;

    for (int m = 0; m < M1; m++) {
        delete[] caches[m];
    }
    delete[] caches;

    for (int m = 0; m < M1; m++) {
        delete[] parameters[m];
    }
    delete[] parameters;

    delete[] rerank_id;
    delete[] verify_id;
    // add
    for (int k = 0; k < K1; k++) {
        delete[] centroid[k];
    }
    delete[] centroid;
    delete[] n_points_cen;

    for (int i = 0; i < rT; i++) {
        delete[] temp_data[i];
    }

    index_fps_r = nullptr;
    index_fps_l = nullptr;
    fp_para = nullptr;
    delete[] temp_data;
    delete[] knn_result;

    while (!PTA_queue->empty()) {
        search_node search_node_f = PTA_queue->top();
        search_node_f.release();
        PTA_queue->pop();
    }
    delete PTA_queue;
    pair_map_data.clear();
}

void PTA::postprocessing() {
    // release
    delete[] C;

    for (int i = 0; i < K1; i++) {
        for (int j = 0; j < M1; j++) {
            fclose(index_fps_all_r[i][j]);
            fclose(index_fps_all_l[i][j]);
            fclose(fp_para_all[i][j]);
        }
    }
    for (int i = 0; i < K1; i++) {
        delete[] index_fps_all_r[i];
        delete[] index_fps_all_l[i];
        delete[] fp_para_all[i];
    }

    delete[] index_fps_all_r;
    delete[] index_fps_all_l;
    delete[] fp_para_all;

    fclose(proj_data_fp);
}

void PTA::build_index(char *kmeans_file, char *proj_data_file) {
    unsigned long int n = 0;
    int m;
    int id_retrived;

    REAL *data1 = new REAL[n_dimensions];
    REAL **centroid = new REAL *[K1];
    for (int iii = 0; iii < K1; ++iii) {
        centroid[iii] = new REAL[n_dimensions];
        memset(centroid[iii], 0, sizeof(REAL) * n_dimensions);
    }

    FILE *proj_ds_fp = fopen(proj_data_file, "rb");
    FILE_CHEACK(proj_ds_fp, proj_data_file);
    float *proj_data = new float[n_points * M1];

    for (int i = 0; i < K1; ++i) {
        std::vector<int> arr;
        clusters.push_back(arr);
    }

    cout << "allocating memory is done " << endl;

    FILE *ds_fp = fopen(data_f, "rb");
    FILE_CHEACK(ds_fp, data_f);
    ifstream fileStream;
    fileStream.open(kmeans_file, ios::in);
    if (fileStream.fail()) {
        cout << "kmeans file doesn't exist" << endl;
        fileStream.close();
        return;
    }

    unsigned int p_id;
    int cluster_id;
    for (n = 0; n < n_points; ++n) {
        fileStream >> p_id;
        if (p_id != n) {
            printf("something wrong with data id \n");
            return;
        }
        fileStream >> cluster_id;
        int ppp = (int) p_id;
        clusters[cluster_id].push_back(ppp);
        fread(data1, sizeof(REAL), n_dimensions, ds_fp);
        fread(&proj_data[n * M1], sizeof(float), M1, proj_ds_fp);

        for (int j = 0; j < n_dimensions; ++j) {
            centroid[cluster_id][j] += data1[j];
        }
        if (n != 0 && n % 1000000 == 0)
            cout << "finished " << n << " points" << endl;
    }
    fileStream.close();
    fclose(ds_fp);
    fclose(proj_ds_fp);
    cout << "finished data partition .." << endl;

    for (int iii = 0; iii < K1; ++iii) {
        int cluster_size = clusters[iii].size();
        cout << "cluster size " << cluster_size << endl;
        Point **sorted_Lists = new Point *[M1];

        for (int i1 = 0; i1 < n_dimensions; ++i1) {
            centroid[iii][i1] /= cluster_size;
        }

        for (m = 0; m < M1; ++m) {
            sorted_Lists[m] = new Point[cluster_size];
        }

        for (n = 0; n < cluster_size; ++n) {
            id_retrived = clusters[iii][n];
            for (int i = 0; i < M1; ++i) {
                sorted_Lists[i][n].id = id_retrived;
                unsigned long int id_retrived_ = id_retrived;
                sorted_Lists[i][n].coordinate = proj_data[id_retrived_ * M1 + i];
            }
        }

        char centroid_file[256];
        sprintf(centroid_file, "%s/%s_%d.centroid", index_f, data_name, iii);
        FILE *cen_fp = fopen(centroid_file, "wb");
        FILE_CHEACK(cen_fp, centroid_file);

        fwrite(centroid[iii], sizeof(float), n_dimensions, cen_fp);
        fwrite(&cluster_size, sizeof(int), 1, cen_fp);
        fclose(cen_fp);

        // sort the Lists, and then save the sorted Lists
        int page_num = PAGESIZE / sizeof(int);
        int *id_sorted = new int[cluster_size];
        int n_blocks = cluster_size / page_num + 1;
        int tail_size = cluster_size % page_num;
        float *para = new float[n_blocks];

        for (m = 0; m < M1; ++m) {
            char index_file[256];
            char para_file[256];
            sprintf(index_file, "%s/%s_%d_%d.index", index_f, data_name, iii, m);
            sprintf(para_file, "%s/%s_%d_%d.para", index_f, data_name, iii, m);

            FILE *index_fp = fopen(index_file, "wb");
            FILE_CHEACK(index_fp, index_file);

            FILE *para_fp = fopen(para_file, "wb");
            FILE_CHEACK(para_fp, para_file);

            sort(sorted_Lists[m], sorted_Lists[m] + cluster_size, qcmps);

            int i = 0;
            float avg_num = 0.0;
            for (int j = 0; j < cluster_size; ++j) {
                id_sorted[j] = sorted_Lists[m][j].id;
                avg_num += sorted_Lists[m][j].coordinate;
                if ((j + 1) % page_num == 0 || j == cluster_size - 1) {
                    if (tail_size != 0 && j == cluster_size - 1) {
                        avg_num = avg_num / tail_size;
                    }
                    else {
                        avg_num = avg_num / page_num;
                    }
                    para[i++] = avg_num;
                    avg_num = 0.0;
                }
            }
            fwrite(id_sorted, sizeof(int), cluster_size, index_fp);
            fwrite(para, sizeof(float), n_blocks, para_fp);

            fclose(index_fp);
            fclose(para_fp);
        }

        for (m = 0; m < M1; ++m) {
            delete[] sorted_Lists[m];
        }
        delete[] sorted_Lists;
        delete[] id_sorted;
        delete[] para;
        printf("finished indexing cluster %d \n", iii);
    }

    delete[] data1;
    delete[] proj_data;
    for (int iii = 0; iii < K1; ++iii) {
        delete[] centroid[iii];
    }
    delete[] centroid;
}

// when reading a node from disk, we add 1 to IO (random IO)
PPoint PTA::binary_search(float *array, REAL key) {
    double q1, q2;
    int low = 0, high = number_of_blocks - 1;
    int mid = 0;
    while (low <= high) {
        mid = (low + high) / 2;
        if (key == array[mid]) {
            // at most case, this won't happen, because the type of them are floating
            return {mid, 0};
        } else if (key < array[mid]) high = mid - 1;
        else low = mid + 1;
    }
    if (key < array[mid]) {
        if (mid == 0) {
            q1 = array[mid] - key;
            return {mid, q1 * q1};
        }
        q1 = key - array[mid - 1];
        q2 = array[mid] - key;
        if (q1 < q2) {
            return {mid - 1, q1 * q1};
        } else {
            return {mid, q2 * q2};
        }
    } else {
        if (mid == number_of_blocks - 1) {
            q1 = key - array[mid];
            return {mid, q1 * q1};
        }
        q1 = key - array[mid];
        q2 = array[mid + 1] - key;
        if (q1 < q2) {
            return {mid, q1 * q1};
        } else {
            return {mid + 1, q2 * q2};
        }
    }
}

unsigned int PTA::binary_search2(float *array, REAL key) {
    REAL q1, q2;
    unsigned int low = 0, high = number_of_blocks - 1;
    unsigned int mid = 0;
    while (low <= high) {
        mid = (low + high) / 2;
        if (key == array[mid]) {
            // at most case, this won't happen, because the type of them are floating
            return mid;
        } else if (key < array[mid]) high = mid - 1;
        else low = mid + 1;
    }
    if (key < array[mid]) {
        if (mid == 0)
            return mid;
        return mid - 1;
    } else {
        if (mid == number_of_blocks - 1)
            return mid;
        return mid + 1;
    }
}

search_node PTA::linear_scan_from_cache(float *parameters, int &para_left_i, int &para_right_i,
                                        float key, int *cache1, int *cache2, int &cache1_i,
                                        int &cache1_i_s, int &cache2_i, int &cache2_i_s, int m_, int page) {
    double q1, q2;

    if (para_left_i >= 0)
        q1 = key - parameters[para_left_i];
    else
        q1 = FLOATMAX;
    if (para_right_i <= number_of_blocks - 1) // on right
        q2 = parameters[para_right_i] - key;
    else
        q2 = FLOATMAX;

    if (para_left_i < 0 && para_right_i >= number_of_blocks)
        return search_node();

    if (q1 < q2 && cache1_i < cache1_i_s) {
        int len_ = 0;
        if (cache1_i + page * cache_size_0 <= cache1_i_s) {
            len_ = page * cache_size_0;
            para_left_i -= page;
        }
        else {
            len_ = cache1_i_s - cache1_i;
            para_left_i -= len_ / cache_size_0;
        }
        search_node search_node1(&cache1[cache1_i], len_, m_, q1 * q1, -1);
        cache1_i = cache1_i + len_;
        return search_node1;
    } else if (q2 <= q1 && cache2_i < cache2_i_s) {
        int len_ = 0;
        if (cache2_i + page * cache_size_0 <= cache2_i_s) {
            len_ = page * cache_size_0;
            para_right_i += page;
        }
        else {
            len_ = cache2_i_s - cache2_i;
            para_right_i += len_ / cache_size_0;
            if (len_ % cache_size_0 != 0) {
                para_right_i++;
            }
        }
        search_node search_node1(&cache2[cache2_i], len_, m_, q2 * q2, 1);
        cache2_i = cache2_i + len_;
        return search_node1;
    }
    return search_node();
}

int PTA::linear_scan_from_file(FILE *fp, int &list_i, int *cache, int c_size,
                               int &cache_i, int direction) {
    int io = 0; // this is sequential IO
    if (direction == 0) {
        if (list_i <= NPS) {
            fseek(fp, 0, SEEK_SET);
            fread(cache, PAGESIZE, list_i, fp);
            cache_i = PAGESIZE * list_i / sizeof(int);
            list_i = 0;
        } else {
            unsigned int diff = list_i - NPS;
            fseek(fp, diff * PAGESIZE, SEEK_SET);
            fread(cache, PAGESIZE, NPS, fp);
            cache_i = c_size;
            list_i = diff;
        }
        io += NPS / 100;
    } else {
        unsigned int diff2 = number_of_blocks - list_i;
        fread(cache, PAGESIZE, NPS, fp);
        if (diff2 <= NPS) {
            if (tail_size == 0)
                cache_i = PAGESIZE * diff2 / sizeof(int);
            else
                cache_i = PAGESIZE * (diff2 - 1) / sizeof(int) + tail_size;
            list_i = number_of_blocks;
        } else {
            cache_i = c_size;
            list_i = list_i + NPS;
        }
        // io++;
        io += NPS / 100;
    }

    return io;
}

void PTA::knn_search_step01(float *proj_query, int &IO, int most_groundtruth_id, int *ori_map_new,
                            char *hamming_step02_id_file, char *hamming_step02_file, float &hit_rate,
                            int *test01, float &time_i, int n_page) {
    timeval start, end;
    gettimeofday(&start, NULL);

    int cluster_id = most_groundtruth_id;

    index_fps_r = index_fps_all_r[cluster_id];
    index_fps_l = index_fps_all_l[cluster_id];
    fp_para = fp_para_all[cluster_id];

    int nn = n_points_cen[cluster_id];
    int remainder = (sizeof(int) * nn) % PAGESIZE;
    int n_blocks = (sizeof(int) * nn) / PAGESIZE;
    if (remainder == 0) {
        number_of_blocks = n_blocks;
        tail_size = 0;
    } else {
        number_of_blocks = n_blocks + 1;
        int tail_len = (sizeof(int) * nn - n_blocks * PAGESIZE);
        tail_size = tail_len / sizeof(int);
    }

    for (int m = 0; m < M1; m++) {
        parameters[m] = new float[number_of_blocks];
    }
    for (int m = 0; m < M1; ++m) {
        fseek(fp_para[m], 0, SEEK_SET);
        fread(parameters[m], sizeof(float), number_of_blocks, fp_para[m]);
        IO += (unsigned int) ceilf(sizeof(float) * number_of_blocks * 1.0f / PAGESIZE / RATIO); // sequential IO

        PPoint block = binary_search(parameters[m], proj_query[m]);

        list_left_i[m] = block.id;
        list_right_i[m] = block.id + 1;
        para_left_i[m] = block.id - 1;
        para_right_i[m] = block.id + 1;

        // read the block in the file with block_id
        fseek(index_fps_r[m], (unsigned long int) PAGESIZE * block.id, SEEK_SET);
        fseek(index_fps_l[m], (unsigned long int) PAGESIZE * block.id, SEEK_SET);
        fread(caches[m], 1, PAGESIZE, index_fps_r[m]);
        IO++;

        // count the number of collision
        if (block.id == number_of_blocks - 1 && tail_size != 0) {
            search_node search_node1(caches[m], tail_size, m, block.coordinate, 0);
            PTA_queue->push(search_node1);
        } else {
            search_node search_node1(caches[m], cache_size_0, m, block.coordinate, 0);
            PTA_queue->push(search_node1);
        }
    }

    int num_ids, m_index, CC;
    int rerank_i = 0, id_retrived = 0, num = 0, n_all = number_of_blocks * M1;
    int *BIDs;
    bool stop_condition = false;

    while (!PTA_queue->empty() && num < n_all && !stop_condition) {
        search_node search_node2 = PTA_queue->top();
        m_index = search_node2.list_m;
        BIDs = search_node2.ids;
        num_ids = search_node2.num_ids;
        for (int i = 0; i < num_ids; ++i) {
            id_retrived = BIDs[i];
            if (++C[id_retrived] == m1) {
                rerank_id[rerank_i++] = id_retrived;
                if (rerank_i >= rT || rerank_i >= nn) {
                    stop_condition = true;
                    break;
                }
            }
        }
        if (stop_condition) {
            break;
        }
        search_node2.release();
        PTA_queue->pop();
        // check the cache, if cache is empty, than load data from file into cache

        if (cache1_i[m_index] >= cache1_i_s[m_index] && list_left_i[m_index] > 0) {
            IO += linear_scan_from_file(index_fps_l[m_index], list_left_i[m_index],
                                        cache1[m_index], cache_size, cache1_i_s[m_index], 0);
            cache1_i[m_index] = 0;
        }
        if (cache2_i[m_index] >= cache2_i_s[m_index] && list_right_i[m_index] <= number_of_blocks - 1) {
            IO += linear_scan_from_file(index_fps_r[m_index], list_right_i[m_index],
                                        cache2[m_index], cache_size, cache2_i_s[m_index], 1);
            cache2_i[m_index] = 0;
        }

        search_node search_node3 = linear_scan_from_cache(parameters[m_index],
                                                          para_left_i[m_index], para_right_i[m_index],
                                                          proj_query[m_index], cache1[m_index], cache2[m_index],
                                                          cache1_i[m_index], cache1_i_s[m_index],
                                                          cache2_i[m_index], cache2_i_s[m_index], m_index, n_page);
        if (search_node3.ids != NULL) {
            PTA_queue->push(search_node3);
        }
        num++;
    }

    for (int i = 0; i < rT; i++) {
        int C_ = rerank_id[i];
        pair_map_data.push_back({C_, ori_map_new[C_]});
    }
    sort(pair_map_data.begin(), pair_map_data.end());

    //verification

    int k_th_id = sT - 1;
    double min_dist = FLOATMAX;

    int ID, mID, tmpID = 0, offset, hit_num = 0;
    float ff, pta_bound;
    int i_ = 0;
    for (const auto& p : pair_map_data) {
        pta_bound = 0;
        ID = p.key;
        mID = p.value;
        // read a data point from dataset in disk
        if (mID - tmpID < cache_size / M1 && i_ != 0) {
            offset = (mID - tmpID) * M1;
            for (int j = 0; j < M1; ++j) {
                data_proj[j] = data_proj_new[offset + j];
            }
            hit_num++;
        }
        else {
            tmpID = mID;
            unsigned long int size2 = sizeof(float) * (unsigned long int) mID * M1;
            fseek(proj_data_fp, size2, SEEK_SET);
            fread(data_proj_new, sizeof(float), cache_size, proj_data_fp);
            IO++;
            offset = (mID - tmpID) * M1;
            for (int j = 0; j < M1; ++j) {
                data_proj[j] = data_proj_new[offset + j];
            }
        }

        for (int j = 0; j < M1; ++j) {
            ff = data_proj[j] - proj_query[j];
            pta_bound += ff * ff;
            temp_data[i_][j] = data_proj[j];
        }

        if (pta_bound < min_dist) {
            knn_result[k_th_id].id = ID;
            knn_result[k_th_id].num= i_;
            knn_result[k_th_id].dist = pta_bound;
            min_dist = 0;
            for (int i = 0; i < sT; ++i) {
                if (knn_result[i].dist > min_dist) {
                    min_dist = knn_result[i].dist;
                    k_th_id = i;
                }
            }
        }
        i_++;
    }

    gettimeofday(&end, NULL);
    time_i = diff_timeval(end, start);

    hit_rate += static_cast<float>(hit_num) / rT;
    // cout << "hit rate is " << static_cast<float>(hit_num) / rT << endl;

    for (int i_ = 0; i_ < sT; ++i_) {
        test01[i_] = knn_result[i_].id;
    }

    ofstream outfile1;
    outfile1.open(hamming_step02_id_file, ios::app);
    if (!outfile1.is_open()) {
        cout << "file open fail" << endl;
        exit(0);
    }

    for (int i_ = 0; i_ < sT; ++i_) {
        outfile1 << test01[i_] << " ";
    }
    outfile1 << endl;
    outfile1.close();

    cout << "the number of verify points is " << sT << endl;

    ofstream outfile2;
    outfile2.open(hamming_step02_file, ios::app);
    if (!outfile2.is_open()) {
        cout << "file open fail" << endl;
        exit(0);
    }

    for (int i_ = 0; i_ < sT; ++i_) {
        for (int j = 0; j < M1; ++j) {
            outfile2 << temp_data[knn_result[i_].num][j] << " ";
        }
        outfile2 << endl;
    }

    outfile2.close();
}

void PTA::knn_search_step02(float *proj_query, REAL *query, int knn, Result *knnResult,
                           int &IO, float &vefiry_time, int *data_step02_id, int alpha,
                           float **data_step02_label, int q, int sT) {
    for (int l = 0; l < knn; ++l) {
        knnResult[l].id = -1;
        knnResult[l].dist = FLOATMAX;
    }
    int k_th_id = knn - 1;
    double min_dist = FLOATMAX;

    cout << "the number of verify points is " << sT << endl;

    timeval start, end;
    gettimeofday(&start, NULL);

    int ID;
    float ff, pta_bound;
    for (int i = 0; i < sT; ++i) {
        pta_bound = 0;
        ID = data_step02_id[i];

        int row = q * sT + i;

        IO++;
        for (int j = 0; j < n_dimensions; ++j) {
            ff = data_step02_label[row][j] - query[j] * alpha;
            pta_bound += ff * ff;
            if (pta_bound >= min_dist) {
                break;
            }
        }

        if (pta_bound < min_dist) {
            knnResult[k_th_id].id = ID;
            knnResult[k_th_id].dist = pta_bound;
            min_dist = 0;
            for (int i_ = 0; i_ < knn; ++i_) {
                if (knnResult[i_].dist > min_dist) {
                    min_dist = knnResult[i_].dist;
                    k_th_id = i_;
                }
            }
        }
    }

    gettimeofday(&end, NULL);
    vefiry_time += diff_timeval(end, start);
}
