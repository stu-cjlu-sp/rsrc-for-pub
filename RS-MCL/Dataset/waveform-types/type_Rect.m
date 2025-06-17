function s = type_Rect(N,fs,A,fc)
t = (1:N)/fs;
phi0 = 2*pi*rand(1) - pi;
s = A*exp(1j*(2*pi*fc*t+phi0));
end