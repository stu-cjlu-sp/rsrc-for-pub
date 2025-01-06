function s = type_Frank(cpp, fs, A, fc, M)
for ii = 1:M
    for jj = 1:M
        phaseCode(ii,jj) = 2*pi/M*(ii-1)*(jj-1);
    end
end
[u,v] = size(phaseCode);
phaseCode = reshape(phaseCode,[1 u*v]);
s = type_BPSK(cpp, fs, A, fc, phaseCode);
end