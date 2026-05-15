% find a number <npca> such that the first <npca> PC vectors have the given
% part of energy
% Yingfan LIU, Nov 10, 2014

function npca = getNpca(X, ratio)
[n d] = size(X);
ratio = ratio * 100;
comatrix = cov(X);
[a b Latent] = pcacov(comatrix);
s = cumsum(Latent);
npca = 0;
for i = 1:d
    if s(i) > ratio
        %npca = i;
        break;
    end
end

if (i<d)
    npca = i;
else
    npca = d;
end
