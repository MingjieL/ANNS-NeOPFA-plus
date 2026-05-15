% In this file, we implement the ground truth. 
% By Yingfan LIU, May 4, 2015
% Department of SEEM, CUHK


addpath utils;
addpath search;
addpath itq;
addpath ANN;
addpath ENN;
addpath BPQ-tree
load 'Corel/data.mat'
gt_matrix = getGroundTruth(X, qs);
save('Corel/data.mat', 'gt_matrix', '-append');

addpath utils;
addpath search;
addpath itq;
addpath ANN;
addpath ENN;
addpath BPQ-tree
load 'Aerial/data.mat'
gt_matrix = getGroundTruth(X, qs);
save('Aerial/data.mat', 'gt_matrix', '-append');

addpath utils;
addpath search;
addpath itq;
addpath ANN;
addpath ENN;
addpath BPQ-tree
load 'Sift/data.mat'
gt_matrix = getGroundTruth(X, qs);
save('Sift/data.mat', 'gt_matrix', '-append');


addpath utils;
addpath search;
addpath itq;
addpath ANN;
addpath ENN;
addpath BPQ-tree
load 'Sift100M/data.mat'
X = single(X);
n = size(X, 1);
qs1000 = randperm(n, 1000);
gt_matrix1000 = getGroundTruth(X, qs1000);



addpath utils;
addpath search;
addpath itq;
addpath ANN;
addpath ENN;
addpath BPQ-tree
load 'Sift100M/data.mat'
X = single(X);
n = size(X, 1);
tic;
gt_matrix1000 = getGroundTruth(X, qs);
toc


nq = length(qs);
ds = zeros(nq, 1);
id = 122285;
pt = X(id, :);
qy = X(qs, :);
ds = edistance(pt, qy);



addpath utils;
addpath search;
addpath itq;
addpath ANN;
addpath ENN;
addpath BPQ-tree
load 'Sift100M/data.mat'
X = single(X);
n = size(X, 1);
nq = 100;
qsfn = '/data/liuyf/datapf/Sift100M/qs.txt';
fid = fopen(qsfn, 'r');
qs = fscanf(fid, '%d');
fclose(fid);
qs = reshape(qs, 2, 1000);
qs = qs';
qs = qs(1:100, 2) + 1;
gtm = getGroundTruth(X, qs);




% the first the query
pid = 40035;
pt = X(pid, :)';


% read the first query from the file
qfn = '/data/liuyf/datapf/Sift100M/query.ds';
d = 128;
fid = fopen(qfn, 'rb');
query = fread(fid, d, 'float32');
fclose(fid);

































