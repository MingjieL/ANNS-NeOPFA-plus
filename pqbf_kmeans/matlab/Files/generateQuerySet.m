% In this file, according to the data set, we randomly generate a specified
% number of queries.
% By Yingfan LIU, on Feb. 20, 2016


function [query, qs] = generateQuerySet(n, d, dfn, qfn, qsfn, nq)
qs = randperm(n, nq);
qs = qs - 1;
qs = sort(qs);
dfid = fopen(dfn, 'rb');
qfid = fopen(qfn, 'wb');
qsfid = fopen(qsfn, 'w');
query = zeros(d, nq);

for i = 1:nq
    qid = qs(i);
    fseek(dfid, qid*d*4, 'bof');
    qv = fread(dfid, d, 'float32');
    reshape(qv, d, 1);
    query(:, i) = qv;
end

for i = 1:nq
    fprintf(qsfid, '%d\t%d\n', i, qs(i));
end

query = single(query);
fwrite(qfid, query, 'float32');

fclose(qsfid);
fclose(dfid);
fclose(qfid);



























