% In this file, we generate the files realted to multi-cluster CKM
% By Yingfan LIU, on Mar. 9, 2016


function mc_generateResultFiles(folder, K1, K2_array, L2_array, n, d, log_fn)
assert(length(K2_array) == length(L2_array));
num = length(K2_array);
% generate the important file names
% * the data file
data_fn = sprintf('%s/data.ds', folder);
% * the trdata file
tr_fn = sprintf('%s/trdata.mat', folder);
load(tr_fn);

trdata = trdata';

b_size = 1000000;

% get the data set name
dsn = getDSname(folder);

beg = cputime;
tic;

% generate the <ckm_res1> and <codes1>
L1 = 1;
% * the <ckm_res1>
ckm_res1 = train_ckmeans_v3(trdata, K1, L1);
% * generate the <codes1>
codes1 = quantization_v1(ckm_res1, trdata, K1, L1);

cpu_cost(1) = cputime - beg;
exc_cost(1) = toc;

fid = fopen(log_fn, 'a+');
for i=1:num
    beg = cputime;
    tic;
    K2 = K2_array(i);
    L2 = L2_array(i);
    % generate <ckm_res2>
    ckm_res2 = mc_train_ckm_v2(trdata, ckm_res1, codes1, K2, L2);
    
    cpu_cost(2) = cputime - beg;
    exc_cost(2) = toc;

    % generate the folder name for this combination of K1, K2 and L2
    ckm_folder = sprintf('%s/forest_%d_%d_%d', folder, K1, K2, L2);
    %if ~exist(ckm_folder, 'dir')
    %    mkdir(ckm_folder);
    %end
    % revised by yfliu, on Sep. 9, 2016
    % the reason is that for large-scale set of code file, we use '-append' mode 
    % to write data to the file, and thus we have to remove existing data when 
    % we newly generate the data for this setting.
    if exist(ckm_folder, 'dir')
        rmdir(ckm_folder, 's');
    end
    mkdir(ckm_folder);
    % write the data into the files 
    % * ckm1
    clus1_fn = sprintf('%s/ckm1.clus', ckm_folder);
    code1_fn = sprintf('%s/codes1.txt', ckm_folder);
    new_write_clusters(clus1_fn, ckm_res1, K1, L1);
    %quantization_v3(ckm_res1, K1, L1, n, d, b_size, data_fn, code1_fn);
    % * ckm2
    clus2_fn = sprintf('%s/ckm2.clus', ckm_folder);
    code2_fn = sprintf('%s/codes2.txt', ckm_folder);
    new_write_clusters(clus2_fn, ckm_res2, K2, L2);
    %mc_quantization_v2(ckm_res1, K1, ckm_res2, K2, L2, n, d, b_size, data_fn, code2_fn);
    
    beg = cputime;
    tic;
    mc_quantization(ckm_res1, K1, ckm_res2, K2, L2, n, d, b_size, data_fn, code1_fn, code2_fn)
    cpu_cost(3) = cputime - beg;
    exc_cost(3) = toc;
    
    % print related information to the log
    fprintf(fid, '-------------------------------------\n');
    fprintf(fid, '%s\t', dsn);
    fprintf(fid, 'K1 = %d\t', K1);
    fprintf(fid, 'K2 = %d\t', K2);
    fprintf(fid, 'L2 = %d\n', L2);
    fprintf(fid, '-------------------------------------\n');
    fprintf(fid, 'ckm1:\t\t%f (%f) seconds.\n', exc_cost(1), cpu_cost(1));
    fprintf(fid, 'ckm2:\t\t%f (%f) seconds.\n', exc_cost(2), cpu_cost(2));
    fprintf(fid, 'Quantization:\t\t%f (%f) seconds.\n', exc_cost(3), cpu_cost(3));
    fprintf(fid, 'Totally:\t%f (%f) seconds.\n', sum(exc_cost), sum(cpu_cost));
end

fclose(fid);



























