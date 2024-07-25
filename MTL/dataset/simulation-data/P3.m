function   y=P3(M,fc,T,fs,SNR)
N=M^2;
for ii=1:N
    phase_shift(ii) =pi/N*(ii-1).^2;
end
%% �����ź�
tb_length = ceil(T*fs/(M*M));      %% ����Ԫ��Ӧ�Ĳ�������Ŀ
PW_length = round(T*fs);                   %% �����ȶ�Ӧ����Ԫ
P3_signal = zeros(1,PW_length);
for kk = 1:PW_length
    Phi(kk)= phase_shift(ceil(kk/tb_length));
    P3_signal(kk) = exp(1i*(2*pi*fc*kk/fs + Phi(kk)));   %% �������������ϸ�����λ�������ȡֵʱ�����ǻ���-pi~+pi֮�������𵴣���˼���pi/4
end
% y=awgn(P3_signal,SNR);
y=P3_signal;
end