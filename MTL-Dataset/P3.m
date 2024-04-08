function   y=P3(M,fc,T,fs,SNR)
N=M^2;
for ii=1:N
    phase_shift(ii) =pi/N*(ii-1).^2;
end
%% 产生信号
tb_length = ceil(T*fs/(M*M));      %% 子码元对应的采样点数目
PW_length = round(T*fs);                   %% 脉冲宽度对应的码元
P3_signal = zeros(1,PW_length);
for kk = 1:PW_length
    Phi(kk)= phase_shift(ceil(kk/tb_length));
    P3_signal(kk) = exp(1i*(2*pi*fc*kk/fs + Phi(kk)));   %% 由于噪声，当严格按照相位跳变矩阵取值时，总是会在-pi~+pi之间来回震荡，因此加上pi/4
end
% y=awgn(P3_signal,SNR);
y=P3_signal;
end