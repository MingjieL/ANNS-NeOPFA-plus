filename = 'D:/桌面/sift/trdata.mat';

filename1 = 'D:/桌面/sift/trdata1.mat';

filename2 = 'D:/桌面/sift/trdata2.mat';

filename3 = 'D:/桌面/sift/trdata3.mat';

filename4 = 'D:/桌面/sift/data.ds';

load(filename3);

disp(trdata(1:3, :));

% n = 1000000;
% d = 128;

% dfid = fopen(filename4, 'rb');
% [data, count] = fread(dfid, n*d, 'float32');
% data = reshape(data, d, num);

% disp(data(:, 1:3));
