% In this file, we generate the cluster infromation into the file
% added by Yingfan LIU, Mar. 24, 2015

function write_clusters(fn, res, sub_dims, R, n, K, L)
fid = fopen(fn, 'wb');

% d : the dimensionality of the original data
% dR : the reduced dimensionality
[d, dR] = size(R);
% wirte the basic parameters
fwrite(fid, n, 'int32');
fwrite(fid, d, 'int32');
fwrite(fid, dR, 'int32');
fwrite(fid, K, 'int32');
fwrite(fid, L, 'int32');
% write <sub_dims> into the file
dim_set = zeros(dR, 1);
for i = 1:L
    dims = sub_dims{i};
    dim_set(dims) = i-1;
end
fwrite(fid, dim_set, 'int32');
% write Rotation Matrix into the file
fwrite(fid, R, 'float32');
% clusters
for i = 1:L
    centers = res(i).ccenters;
    centers = centers';
    fwrite(fid, centers, 'float32');
end
% PCA orders
orders = zeros(K, 1);
for i = 1:L
    centers = res(i).ccenters;
    [cens, ids] = LinearOrder(centers);
    orders(ids) = 0:(K-1);
    fwrite(fid, orders, 'int32');
end

fclose(fid);









