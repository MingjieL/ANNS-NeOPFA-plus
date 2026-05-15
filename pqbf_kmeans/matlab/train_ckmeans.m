% rewrite the training model function
% Modified By Yingfan LIU, Nov 9, 2014

% Inputs:
%   trdata: training data.
%   X : the original data set
%   K : the number of clusters in each subspace
%   L : the number of subspaces

% Output:
%   res : the format as the output of0 ProClus 
%   sub_dims : the result of dimension partition
%   R : the rotation matrix as a pre-step of ck-means   

function [res, sub_dims, R] = train_ckmeans(trdata, X, K, L, ckm_mode)
h_ckmeans = K;
m_ckmeans = L;
niter = 100;        % the number of iterations is set as 100 by default
mu = mean(trdata, 2);

% PCA dimensionality reduction as a pre-processing to speed-up
% training of the quantization methods.

npca = size(trdata, 1) - mod(size(trdata, 1), m_ckmeans);
%ratio = 1;
%npca = getNpca(trdata', ratio);
%npca = ceil(npca/L)*L;
%npca = min(npca, size(trdata, 1) - mod(size(trdata, 1), m_ckmeans));

if (npca == size(trdata, 1))
    pc = eye(size(trdata, 1));
else
    [pc, l] = eigs(cov(double(bsxfun(@minus, trdata, mu)')), npca);
end
trdata2 = pc' * bsxfun(@minus, trdata, mu);
fprintf('trdata2 created, size(trdata2) = (%d, %.1e).\n', ...
          size(trdata2, 1), size(trdata2, 2));


% Run the actual training of the models.

%ckmeans_init = 'natural';
ckmeans_init = ckm_mode;
fprintf('ckmeans init: "%s"\n', ckmeans_init);
model = ckmeans(trdata2, m_ckmeans, h_ckmeans, niter, ckmeans_init);


% Revert the effect of PCA dimensionality reduction. This is done to
% only keep the model around, and throw away pc and mu.
R = pc * model.R;

model.R = pc * model.R;
Rmu = model.R' * mu;
len0 = 1 + cumsum([0; model.len(1:end-1)]);
len1 = cumsum(model.len);
for i = 1:model.m
    % Add mu back
    model.centers{i} = bsxfun(@plus, model.centers{i}, Rmu(len0(i):len1(i)));
end
model.preprocess.pc = pc;
model.preprocess.mu = mu;


% *************** modified by Yingfan LIU ************** %
% on Nov 10, 2014

% construct the sturcture needed in my methods
% generate results of dimension partition 
sub_dims = cell(1,L);
for i = 1:L
    sub_dims{i} = len0(i):len1(i);
end

RX = model.R' * X;
% generate the results for PQ-File format
for i = 1:L
    %fprintf('i=%d\n', i);
    dims = sub_dims{i};
    data = RX(dims, :)';
    % the centers
    cens = model.centers{i}';
    %ccenters = LinearOrder(cens);
    ccenters = cens;
    % reformat the clustering results
    [clusters idxp maxvs minvs] = reformat(data, ccenters);
    res(i).ccenters = ccenters;
    res(i).clusters = clusters;
    res(i).idxp = idxp;
    res(i).maxvs = maxvs;
    res(i).minvs = minvs;
    res(i).ckm_mode = ckm_mode;
end


% recall that X is n*d format and ccenters have row vectors
% however, X and ccenters in CK-Means codes are both colmun vectors
function [clusters idxp maxvs minvs] = reformat(X, ccenters)
[K d] = size(ccenters);
idxp = euc_nn_mex(ccenters',X'); 
clusters = cell(1,K);
maxvs = zeros(K,d);
minvs = zeros(K,d);

for i = 1:K
    ii = find(idxp == i);
    clusters{1,i} = ii;
    maxvs(i,:) = max(X(ii,:));
    minvs(i,:) = min(X(ii,:));
end








