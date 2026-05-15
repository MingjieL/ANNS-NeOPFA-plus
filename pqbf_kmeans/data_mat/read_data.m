filename = 'D:/桌面/sift/sift_base.fvecs';
trdata = fvecs_read (filename);
trdata = trdata';

% filename = 'D:/桌面/sift/sift_groundtruth.ivecs';
% v = ivecs_read (filename)

% disp(v)

save('D:/桌面/sift/trdata2.mat', 'trdata');

