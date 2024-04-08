function   y=P4(M,fc,T,fs,SNR)
N=M*M;
for jj=1:N
    phase_shift(jj) = (pi/N)*(jj-1)^2-pi*(jj-1);
    
end
%% �����ź�
tb_length = ceil(T*fs/(M*M));      %% ����Ԫ��Ӧ�Ĳ�������Ŀ
PW_length = round(T*fs);                   %% �����ȶ�Ӧ����Ԫ

P4_signal = zeros(1,PW_length);
for kk = 1:PW_length
    Phi(kk)= phase_shift(ceil(kk/tb_length));
    P4_signal(kk) = exp(1i*(2*pi*fc*kk/fs + Phi(kk)));   %% �������������ϸ�����λ�������ȡֵʱ�����ǻ���-pi~+pi֮�������𵴣���˼���pi/4
end
% y=awgn(P4_signal,SNR);
y=P4_signal;
end