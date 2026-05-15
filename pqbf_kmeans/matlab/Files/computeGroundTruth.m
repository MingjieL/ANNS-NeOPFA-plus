% In this file, we implement a function to get the ground truth, which
% obtain the data from the data.ds.
% By Yingfan LIU, on Feb. 29, 2016


function gt_matrix = computeGroundTruth(dfn, n, d, qfn, nq, NK)
dfid = fopen(dfn, 'rb');
qfid = fopen(qfn, 'rb');
step = 1000000;
queries = fread(qfid, [d, nq], 'float32');
gt_matrix = zeros(nq, NK);
% the result structure
dists = [];
for i=1:nq
    fprintf('i=%d\n', i);
    rem = n;
    query = queries(:, i);
    dists = [];
    iter = ceil(n / step);
    Q = repmat(query, 1, step);
    fseek(dfid, 0, 'bof');
    for j=1:iter
        if step > rem
            num = rem;
            Q = Q(:, 1:num);
        else
            num = step;
        end
        M = fread(dfid, [d, num], 'float32');
        temp = sum((Q-M).^2, 1);
        dists = [dists, temp];
        dists = sort(dists);
        dists = dists(1:NK);
        rem = rem - num;
    end
    dists = sqrt(dists);
    gt_matrix(i, :) = dists;
end

fclose(dfid);
fclose(qfid);























