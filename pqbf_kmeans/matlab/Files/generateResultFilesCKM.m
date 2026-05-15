% In this file, we generate the .clus and codes.txt file according to the
% inputs for a single pair of K and L
% Yingfan LIU, on Mar. 2, 2016

function [cpu_cost, exc_cost] = generateResultFilesCKM(K, L, trdata, clus_fn, code_fn, n, d, b_size, data_fn)
cpu_cost = zeros(2, 1);
exc_cost = zeros(2, 1);
% generate the CKM data and write the data into the corresponding files
beg = cputime;
tic;
ckm_res = train_ckmeans_v3(trdata', K, L);
cpu_cost(1) = cputime - beg;
exc_cost(1) = toc;
new_write_clusters(clus_fn, ckm_res, K, L);
% generate the codes data and write the data into the file
beg = cputime;
tic;
quantization_v3(ckm_res, K, L, n, d, b_size, data_fn, code_fn);
cpu_cost(2) = cputime - beg;
exc_cost(2) = toc;




































