% In this file, we generate the residual vectors given Odata and ckm_res
% By Yingfan LIU, on April 25, 2016

% recall that Odata are column vectors while centers are row vectors

function rd_data = obtainResidualData(Odata, codes, ckm_res, L)
appro_data = [];
ccenters = ckm_res.ccenters;

% * generate the approximation vectors
for i=1:L
    centers = ccenters{i};
    ids = codes(:, i);
    appro_data = [appro_data centers(ids, :)];
end
appro_data = appro_data';
rd_data = Odata - appro_data;





































