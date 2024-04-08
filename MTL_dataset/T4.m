function   y=T4(fc,t,n,T,F,SNR)
pha=[];
pha=mod(2*pi/n*ceil(n*F*t.^2/(2*T)-n*F*t/2),2*pi);
y=exp(1i*(2*pi*fc*t+pha));
% y=awgn(y,SNR);
end