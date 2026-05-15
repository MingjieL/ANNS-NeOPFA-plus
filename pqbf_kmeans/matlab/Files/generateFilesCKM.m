% In this file, we generate the files given a set of pairs of K and L. 
% By Yingfan LIU, on Mar. 2, 2016


function generateFilesCKM(K_array, L_array, folder, n, d, log_fn)
% check the validity of the input parameters
num = length(K_array);
assert(num == length(L_array));
% generate tr_fn and load trdata into the workshop
tr_fn = strcat(folder, '/trdata.mat');
load(tr_fn);
% generate data_fn
data_fn = strcat(folder, '/data.ds');
% set b_size as 1 million by default
b_size = 1000000;
% get the data set name
dsn = getDSname(folder);
% open the log file 
fid = fopen(log_fn, 'a+');
for i=1:num
    fprintf(fid, '----------------------------------------------------\r\n');
    fprintf(fid, 'From %s ', datestr(now, 31)); 
    % set the parameters
    K = K_array(i);
    L = L_array(i);
    fprintf('K = %d, L = %d\n', K, L);
    % generate the sub folder name
    sfn = strcat(folder, '/tree_');
    sfn = strcat(sfn, num2str(K));
    sfn = strcat(sfn, '_');
    sfn = strcat(sfn, num2str(L));
    %if ~exist(ckm_folder, 'dir')
    %    mkdir(ckm_folder);
    %end
    % revised by yfliu, on Sep. 9, 2016
    % the reason is that for large-scale set of code file, we use '-append' mode 
    % to write data to the file, and thus we have to remove existing data when 
    % we newly generate the data for this setting.
    if exist(sfn, 'dir')
        rmdir(sfn, 's');
    end
    mkdir(sfn);
    % generate .clus file name
    clus_fn = strcat(sfn, '/ckm.clus');
    % generate codes file name
    code_fn = strcat(sfn, '/codes.txt');
    % handle the files for this pair of K and L
    [cpu_cost, exc_cost] = generateResultFilesCKM(K, L, trdata, clus_fn, code_fn, n, d, b_size, data_fn);
    fprintf(fid, 'to %s \r\n', datestr(now, 31));
    % print related information to the log
    fprintf(fid, '%s\t', dsn);
    fprintf(fid, 'K = %d\t', K);
    fprintf(fid, 'L = %d\r\n', L);
    fprintf(fid, 'ckm:\t\t%f (%f) seconds.\r\n', exc_cost(1), cpu_cost(1));
    fprintf(fid, 'quantization:\t%f (%f) seconds.\r\n', exc_cost(2), cpu_cost(2));
    fprintf(fid, 'Totally:\t%f (%f) seconds.\r\n', sum(exc_cost), sum(cpu_cost));
end
fclose(fid);





































