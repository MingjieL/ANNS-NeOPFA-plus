% In this file, we re-define the CK-Means procedure to reduce the memory
% space possibly produced. 
% By Yingfan LIU, Jan. 6, 2015
% Department of SEEM, CUHK


% Inputs:
%   trdata: training data
%   K : the number of clusters in each subspace
%   L : the number of subspaces


% Output:
%   res : the format as the output of CK-Means 
%   sub_dims : the result of dimension partition
%   R : the rotation matrix as a pre-step of ck-means   

function ckm_res = train_ckmeans_v3(trdata, K, L)
h_ckmeans = K;
m_ckmeans = L;
niter = 100;        % the number of iterations is set as 100 by default
mu = mean(trdata, 2);

% settings, added by Yingfan LIU 
ckm_mode = 'natural';
er = 1;

% PCA dimensionality reduction as a pre-processing to speed-up
% training of the quantization methods.

%npca = size(trdata, 1) - mod(size(trdata, 1), m_ckmeans);
ratio = er;
npca = getNpca(trdata', ratio);
npca = ceil(npca/L)*L;
npca = min(npca, size(trdata, 1) - mod(size(trdata, 1), m_ckmeans));

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

% generate the result structure.
% <sub_dims>
sub_dims = cell(1,L);
for i = 1:L
    sub_dims{i} = len0(i):len1(i);
end
ckm_res.sub_dims = sub_dims;
% <R>
ckm_res.R = R;
% <centers>
centers = cell(1,L);
for i = 1:L
    cens = model.centers{i}';
    centers{i} = cens;
end
ckm_res.ccenters = centers;
% <quantization error>
ckm_res.tr_error = model.obj(end);







