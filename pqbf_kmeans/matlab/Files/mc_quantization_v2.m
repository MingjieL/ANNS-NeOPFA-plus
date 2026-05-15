% In this file, we generate the quantizaiton results for ck_means with
% residual data.
% By Yingfan LIU, Mar. 09, 2016


function mc_quantization_v2(ckm_res1, K1, ckm_res2, K2, L2, n, d, b_size, dfn, cfn2)
niter = ceil(n / b_size);
dfid = fopen(dfn, 'rb');
%cfid = fopen(cfn, 'w');
L1 = 1;
R1 = ckm_res1.R; % d * dR
R1 = R1';        % dR * d
centers1 = ckm_res1.ccenters{1}; % K1 * dR
centers1 = centers1';           % dR * K1

if exist(cfn1)
    delete(cfn1);
end

if exist(cfn2)
    delete(cfn2);
end

for i = 1:niter
    ibeg = (i-1) * b_size;
    if i*b_size > n
        num = n - (i-1)*b_size;
        iend = n;
    else
        num = b_size;
        iend = i*b_size - 1;
    end
    ids = (ibeg:iend)';
    [data, count] = fread(dfid, num*d, 'float32');
    data = reshape(data, d, num);
    % generate <codes1>
    codes1 = quantization_v1(ckm_res1, data, K1, L1);
    %write_code_to_txt(cfn1, codes1, ids);
    %generate <codes2>
    Rdata = R1 * data;
    Rdata = Rdata - centers1(:, codes1);
    codes2 = quantization_v1(ckm_res2, Rdata, K2, L2);
    write_code_to_txt(cfn2, codes2, ids);
    fprintf('%d, %d\n', i*b_size, count);
end

fclose(dfid);