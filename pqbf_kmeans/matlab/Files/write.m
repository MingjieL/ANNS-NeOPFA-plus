% In this file, we write the data into the file
% Created by Yingfan LIU, on September 2, 2015

addpath Files
load 'Corel/data16'
dfn = 'Corel/data16.ds';
qfn = 'Corel/query16.ds';
writeDataToFile(X, qs, dfn, qfn);


addpath Files
load 'Sift/data30'
dfn = 'Sift/data30.ds';
qfn = 'Sift/query30.ds';
writeDataToFile(X, qs, dfn, qfn);

addpath Files
addpath cluster
addpath search
addpath utils
addpath HB
load 'Sift/data30'
K = 512;
L = 8;
trdata = X;
fn = 'Sift/clus30.info';
[res, sub_dims, R] = train_ckmeans(trdata, K, L);
new_write_clusters(fn, res, sub_dims, R, K, L);
[codes, MBR, HB] = coding(sub_dims, R, res, X);


addpath Files
load 'Sift/data'
dfn = 'Sift/data.ds';
qfn = 'Sift/query.ds';
writeDataToFile(X, qs, dfn, qfn);


addpath Files
load 'Corel/data16'
K = 128;
L = 4;
trdata = X;
fn = 'Corel/clus16.info';
[res, sub_dims, R] = train_ckmeans(trdata, K, L);
new_write_clusters(fn, res, sub_dims, R, K, L);
[codes, MBR, HB] = coding(sub_dims, R, res, X);

qid = 1;
query = X(qs(qid), :);
qR = query * R;
book1 = computeLBbookAQD(qR, res, sub_dims, K, L)';
book2 = computeLBbookMBR(qR, MBR, sub_dims, K, L)';
book3 = computeLBbookHB(qR, res, HB, sub_dims, K, L)';



addpath Files
addpath cluster
addpath search
addpath utils
addpath HB
load 'Sift/data30'
K = 512;
L = 8;
trdata = X;
fn = 'Sift/clus30.info';
[res, sub_dims, R] = train_ckmeans(trdata, K, L);
new_write_clusters(fn, res, sub_dims, R, K, L);
[codes, MBR, HB] = coding(sub_dims, R, res, X);





addpath Files
addpath cluster
addpath search
addpath utils
addpath HB
load 'Corel/data'
K = 256;
L = 8;
trdata = X;
fn = 'Corel/ckm_256_8.clus';
[res, sub_dims, R] = train_ckmeans(trdata, K, L);
new_write_clusters(fn, res, sub_dims, R, K, L);
dfn = 'Corel/data.ds';
qfn = 'Corel/query.ds';
writeDataToFile(X, qs, dfn, qfn);


sid = 4;
centers = res{sid};
[cens, orders, ids] = loZC(centers);
cens = centers(ids, :);









