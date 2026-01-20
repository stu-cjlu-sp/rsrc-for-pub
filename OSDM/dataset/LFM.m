function  y=LFM(fc,t,K,SNR)
y=exp(1i*(2*pi*fc*t+pi*K*t.^2));
% y=awgn(y,SNR);
% plot(real(y));

end
