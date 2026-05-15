% for Text2image

% task 1 : forest data when K = 64
clear;

addpath utils;
addpath search;
addpath Files;
n = 999999900;
d = 200;
folder = '/root/hdd-data/Text2image';
log_fn = '/root/hdd-data/Text2image/log.txt';
K1 = 64;
K2_array = [256];
L2_array = [8];
tic
t1 = clock;
mc_generateResultFiles(folder, K1, K2_array, L2_array, n, d, log_fn);
t2 = clock;
construction_time2 = toc
construction_time = etime(t2,t1)

construction_time_file = '/root/hdd-data/Text2image/construction_time.txt';
fileID = fopen(construction_time_file, 'a+');
fprintf(fileID, '%f\n', construction_time);
fclose(fileID);