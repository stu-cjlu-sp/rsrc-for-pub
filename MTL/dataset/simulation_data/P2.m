function   y=P2(M,fc,T,fs,SNR)
phase_shift = zeros(M,M);
%% 产生相位变化矩阵
for ii=1:M
    for jj=1:M
        phase_shift(jj,ii) = (-pi/(2*M))*(2*ii-1-M)*(2*jj-M-1);
    end
end
phase_shift_array = reshape(phase_shift',1,M^2);
%% 产生信号
tb_length = ceil(T*fs/(M*M));      %% 子码元对应的采样点数目
PW_length = round(T*fs);                   %% 脉冲宽度对应的码元
P2_signal = zeros(1,PW_length);
for kk = 1:PW_length
    Phi(kk)= phase_shift_array(ceil(kk/tb_length));
    P2_signal(kk) = exp(1i*(2*pi*fc*kk/fs + Phi(kk)+pi/4));   %% 由于噪声，当严格按照相位跳变矩阵取值时，总是会在-pi~+pi之间来回震荡，因此加上pi/4
end
% y=awgn(P2_signal,SNR);
y=P2_signal;
end

