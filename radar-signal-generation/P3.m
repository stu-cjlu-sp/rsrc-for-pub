function s = type_P3(cpp, fs, A, fc, p)
for ii = 1:p
    phaseCode(ii) = pi/p*(ii-1)^2;
end
s = BPSK(cpp, fs, A, fc, phaseCode);
end