% In this file, we generate the cluster infromation into the file
% added by Yingfan LIU, Mar. 24, 2015

function new_write_clusters(fn, ckm_res, K, L)
fid = fopen(fn, 'wb');

sub_dims = ckm_res.sub_dims;
R = ckm_res.R;
res = ckm_res.ccenters;

% d : the dimensionality of the original data
% dR : the reduced dimensionality
[d, dR] = size(R);
% wirte the basic parameters
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
    centers = res{i};
    centers = centers';
    fwrite(fid, centers, 'float32');
end


fclose(fid);









