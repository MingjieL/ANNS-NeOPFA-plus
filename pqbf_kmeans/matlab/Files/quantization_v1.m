% In this file, we generate the codes according to the results of CK-Means.
% The original data is just given. 
% By Yingfan LIU, on Jan. 8, 2016.

% X : column vectors

function codes = quantization_v1(ckm_res, X, K, L)
n = size(X, 2);
sub_dims = ckm_res.sub_dims;
R = ckm_res.R;
ccenters = ckm_res.ccenters;

RX = R' * X;
codes = zeros(n, L);
% generate the results for PQ-File format
for i = 1:L
    %fprintf('i=%d\n', i);
    dims = sub_dims{i};
    data = RX(dims, :)';
    % the centers
    cens = ccenters{i};
    % reformat the clustering results
    idxp = reformat(data, cens);
    codes(:, i) = idxp;
end

codes = int32(codes);


% recall that X is n*d format and ccenters have row vectors
% however, X and ccenters in CK-Means codes are both colmun vectors
function idxp = reformat(X, ccenters)
idxp = euc_nn_mex(ccenters',X'); 