%% load data
% trdata = load('/projects/NNSGroup/MingjieLi/Data/deep1b/deep1b_sample_5000000.txt');

% save /projects/NNSGroup/MingjieLi/Data/deep1b/deep1b trdata -v7.3

% fid = fopen('/data/deep1b/base/data.ds', 'rb');

% M = 1000000000; % 根据实际情况修改
% N = 96;  % 根据实际情况修改
% trdata = fread(fid, [M, N], 'float');

% fclose(fid);

trdata = load('/data/deep1b/base/deep1b_sample_10000000.txt');

save /data/deep1b/base/trdata trdata -v7.3