function   y=P2(M,fc,T,fs,SNR)
phase_shift = zeros(M,M);
%% ������λ�仯����
for ii=1:M
    for jj=1:M
        phase_shift(jj,ii) = (-pi/(2*M))*(2*ii-1-M)*(2*jj-M-1);
    end
end
phase_shift_array = reshape(phase_shift',1,M^2);
%% �����ź�
tb_length = ceil(T*fs/(M*M));      %% ����Ԫ��Ӧ�Ĳ�������Ŀ
PW_length = round(T*fs);                   %% �����ȶ�Ӧ����Ԫ
P2_signal = zeros(1,PW_length);
for kk = 1:PW_length
    Phi(kk)= phase_shift_array(ceil(kk/tb_length));
    P2_signal(kk) = exp(1i*(2*pi*fc*kk/fs + Phi(kk)+pi/4));   %% �������������ϸ�����λ�������ȡֵʱ�����ǻ���-pi~+pi֮�������𵴣���˼���pi/4
end
% y=awgn(P2_signal,SNR);
y=P2_signal;
end

