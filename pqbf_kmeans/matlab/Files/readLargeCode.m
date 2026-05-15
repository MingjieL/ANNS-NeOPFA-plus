% In this function, we implement a function that reads a large number of codes from .txt file
% By Yingfan LIU, on Sep. 7, 2016



function codes = readLargeCode(fn, n, L, nstep)
assert(nstep > 0);
assert(L>0);
assert(n>0);

codes = [];
niter = ceil(n / nstep);
rm = n;
fid = fopen(fn, 'r');

for i = 1:niter
	fprintf('Iteration %d : ', i);
	% set the number of codes to be read in this iteration
	if rm < nstep
		num = rm;
	else
		num = nstep;
	end
	fprintf('%d\n', num);

	% read the codes from the file
	data = fscanf(fid, '%d', [L+1, num]);
	data = data';	
	data = data(:, 2:(L+1));
	% recover the codes by adding 1 to each element
	data = data + 1;
	% add this part to the final structure
	codes = [codes;data];
	rm = rm - num;
end

fclose(fid);


























