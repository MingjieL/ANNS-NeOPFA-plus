% In this file, we generate the ckm file for PQ-ENN to do some testings. 
% By Yingfan LIU, on April 30, 2016


function generateCKMenn(folder, Ks, Ls)
assert(length(Ks) == length(Ls));
num = length(Ks);
fn = sprintf('%s/data.mat', folder);
load(fn);


for i = 1:num
    K = Ks(i);
    L = Ls(i);
    ckmfolder = sprintf('%s/ckm_%d_%d', folder, K, L);
    if ~exist(ckmfolder, 'dir')
        mkdir(ckmfolder);
    end
    fn = sprintf('%s/ckm.clus', ckmfolder);
    ckm_res = train_ckmeans_v3(X', K, L);
    new_write_clusters(fn, ckm_res, K, L);
end







































