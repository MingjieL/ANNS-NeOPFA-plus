% get the name of the data set from the folder name
function dsn = getDSname(str)
len = length(str);
tmp = [];
for i = len:-1:1
    if str(i) ~= '/'
        tmp = [tmp str(i)];
    else
        break;
    end
end

len = length(tmp);
dsn = [];
for i = 1:len
    dsn = [dsn tmp(len-i+1)];
end