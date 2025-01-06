function s = type_BPSK(cpp, fs, A, fc, phaseCode) % ok
Ts = 1/fs;
Tc = cpp/fc;
t = 0:Ts:Tc-Ts;
for k = 1:length(phaseCode)
    s1(:,k) = A*exp(1j*(2*pi*fc*t+phaseCode(k)));
end
[u,v] = size(s1);
s = reshape(s1,[1, u*v]);
end
