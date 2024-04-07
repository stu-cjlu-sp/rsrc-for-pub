function s = P2(cpp, fs, A, fc, M)
for ii = 1:M
    for jj = 1:M
        phaseCode(ii,jj) = -pi/(2*M)*(2*ii-1-M)*(2*jj-1-M);
    end
end
[u,v] = size(phaseCode);
phaseCode = reshape(phaseCode,[1 u*v]);
s = BPSK(cpp, fs, A, fc, phaseCode);
end