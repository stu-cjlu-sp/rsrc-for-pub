function s = type_T2(fs, A, fc, Nps,Ng)
Ts = 1/fs;
Tc = 4./fc;
t = 0:Ts:Tc-Ts;
pw = Tc*Ng;
for jj = 1:Ng-1
    phaseCode(jj,:) = mod(2*pi/Nps*floor((Ng*t-jj*pw)*(2*jj-Ng+1)/pw*Nps/2),2*pi);
end
[u,v] = size(phaseCode);
phaseCode = reshape(phaseCode',[1 u*v]);
s = type_BPSK(2, fs, A, fc, phaseCode);
end