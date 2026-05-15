% In this file, we implement the training method for IVFADC
% i.e. First, the data set is divided into several clusters
% Second, perform CKM on the residual data
% By Yingfan LIU, on Mar. 9, 2016

% trdata : column vectors

function ckm_res2 = mc_train_ckm_v2(trdata, ckm_res1, codes1, K2, L2)
% generate the outer ckm result
%L1 = 1;
% * the <ckm_res1>
%ckm_res1 = train_ckmeans_v3(trdata, K1, L1);
% * generate the <codes1>
%codes1 = quantization_v1(ckm_res1, trdata, K1, L1);

% the structure in <ckm_res1>
assert(length(ckm_res1.ccenters) == 1);
%L1 = 1;
R1 = ckm_res1.R;
R1 = R1'; % dR * d
centers1 = ckm_res1.ccenters{1};    % K1 * dR
centers1 = centers1';              % dR * K1

%K1 = size(centers1, 2);

% generate the CKM results for the residual data
% * generate the residual data
Rdata = R1 * trdata; % dR * n
Rdata = Rdata - centers1(:, codes1);
% * the <ckm_res2>
ckm_res2 = train_ckmeans_v3(Rdata, K2, L2);