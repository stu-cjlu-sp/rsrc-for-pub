function s = type_P4(cpp, fs, A, fc, p)
for ii = 1:p
    phaseCode(ii) = pi/p*(ii-1)^2-pi*(ii-1);
end
s = type_BPSK(cpp, fs, A, fc, phaseCode);
end