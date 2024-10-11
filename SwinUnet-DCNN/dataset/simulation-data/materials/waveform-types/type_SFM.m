function s = type_SFM(NumberSamples,fs,A,fc,f_dev) % ok
% pw = NumberSamples/fs;%脉宽
% f_dev= 
% Tc = NumberSamples/fc;
% T =  NumberSamples / fs;
t = 0:1/fs:(NumberSamples-1)/fs;
f0 = 100e3;%调制频率或者90e3
phi0 = 2*pi*rand(1) - pi;
% phi0 = rand() * 2 * pi;
% f0 = 50;             % 基频（Hz）是中心频率
% f_dev = 200;          % 频率偏移（Hz）#调制系数

% s = A*exp(1j * 2*pi * (fc * t + (f_dev/(2*pi*f0)) * sin(2*pi*f0*t))+phi0);
s = A * exp(1j * (2*pi * (fc * t + (f_dev/(2*pi*f0)) * sin(2*pi*f0*t)) + phi0));

end