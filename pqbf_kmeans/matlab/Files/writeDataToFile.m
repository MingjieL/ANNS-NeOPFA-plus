% In this file, we create the function that write the <data> and <query>
% into the files.
% Created by Yingfan LIU, on September 2, 2015

function writeDataToFile(X, qs, data_file, query_file)
dfid = fopen(data_file, 'w');
qfid = fopen(query_file, 'w');
data = X';
query = X(qs, :);
query = query';
count1 = fwrite(dfid, data, 'float32');
count2 = fwrite(qfid, query, 'float32');
fclose(dfid);
fclose(qfid);