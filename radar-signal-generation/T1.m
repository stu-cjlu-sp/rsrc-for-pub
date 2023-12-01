function s = type_T1(fs, A, fc, Nps,Ng) % ok
Ts = 1/fs;
Tc = 1/fc;
t = 0:Ts:Tc-Ts;
pw = Tc*Ng;%信号脉冲间隔

for jj = 1:Ng-1
    phaseCode(jj,:) = mod(2*pi/Nps*floor((Ng*t-jj*pw)*jj*Nps/pw),2*pi);
end
[u,v] = size(phaseCode);
phaseCode = reshape(phaseCode',[1 u*v]);
s = BPSK(2, fs, A, fc, phaseCode);
end

