function s = T3(NumberSamples, fs, A, fc, Nps,B)
Ts = 1/fs;
Tc = 1/fc;
pw = NumberSamples/fs;%调制周期
t = 0:Ts:pw-Ts;
phaseCode = mod(2*pi/Nps*floor(Nps*B*t.^2/(2*pw)),2*pi);
s = A*exp(1j*(2*pi*fc*t+phaseCode));
end