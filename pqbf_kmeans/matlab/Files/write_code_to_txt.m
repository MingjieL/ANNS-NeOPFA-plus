% In this file, we write the codes to the txt file.
% By Yingfan LIU, Feb. 13, 2016


function write_code_to_txt(fn, codes, ids)
codes = codes - 1;
fid = fopen(fn, 'a+');
[n,d] = size(codes);

for i=1:n
    fprintf(fid, '%d ', ids(i));
    for j=1:d
        fprintf(fid, '%d ', codes(i, j));
    end
    fprintf(fid, '\r\n');
end

%data = [(0:n-1)' codes];
%data = data';
%fwrite(fid, data, 'int32');
fclose(fid);