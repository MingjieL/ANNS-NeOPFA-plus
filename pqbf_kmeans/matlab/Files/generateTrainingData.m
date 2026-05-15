% In this file, we generate the training data from the .ds file into .mat
% file.
% By Yingfan Feb. 22, 2016

function [trdata, oids, nids] = generateTrainingData(dfn, n, d, tfn, r)
assert((r>0) && (r<=1.0));
trdata = [];
oids = [];
nids = [];
step = 1000000;
sid = 0;
count = 0;
fid = fopen(dfn, 'rb');
while sid < n
    data = fread(fid, step*d, 'float32');
    [a, b] = size(data);
    num = a * b / d;
    data = reshape(data, d, num);
    sn = ceil(num * r);
    ids = randperm(num, sn);
    ids = sort(ids);
    sample = data(:, ids);
    sample = single(sample);
    trdata = [trdata sample];
    spid = randi(sn, 1);
    oids = [oids; sid + ids(spid)];
    nids = [nids; count + spid];
    sid = sid + num;
    count = count + sn;
    fprintf('sid = %d, count = %d\n', sid, count);
end

fclose(fid);
trdata = trdata';
save(tfn, 'trdata', '-v7.3');






























