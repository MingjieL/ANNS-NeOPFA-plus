% In this file, we obtain the ground truth for each data set.
% By Yingfan LIU, on Feb. 29, 2016


% for Sift100M
dfn = '/data/liuyf/datapf/Sift100M1/data.ds';
qfn = '/data/liuyf/datapf/Sift100M1/query.ds';
rfn = '/data/liuyf/datapf/Sift100M1/groundtruth.txt';
n = 1e8;
d = 128;
nq = 100;
nk = 100;
gt_matrix = computeGroundTruth(dfn, n, d, qfn, nq, nk);
fid = fopen(rfn, 'r');
fprintf(fid, '%d %d\n', nq, nk);
for i=1:nq
    fprintf(fid, '%d ', i);
    for j=1:nk
        fprintf(fid, '%f ', gt_matrix(i, j));
    end
    fprintf(fid, '\n');
end
fclose(fid);



% for Sift100M
dfn = '/data/liuyf/datapf/Sift100M/data.ds';
qfn = '/data/liuyf/datapf/Sift100M/query.ds';
rfn = '/data/liuyf/datapf/Sift100M/groundtruth21.txt';
n = 1e8;
d = 128;
nq = 100;
nk = 100;
gt_matrix = computeGroundTruth(dfn, n, d, qfn, nq, nk);
fid = fopen(rfn, 'r');
fprintf(fid, '%d %d\n', nq, nk);
for i=1:nq
    fprintf(fid, '%d ', i);
    for j=1:nk
        fprintf(fid, '%f ', gt_matrix(i, j));
    end
    fprintf(fid, '\n');
end
fclose(fid);






















