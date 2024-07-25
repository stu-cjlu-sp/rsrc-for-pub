function   y=P1(M,fc,T,fs,SNR)
for ii=1:M
    for jj=1:M
        phase_shift(jj,ii) = (-pi/M)*(M-(2*jj-1))*((jj-1)*M+(ii-1));
    end
end
phase_shift_array = reshape(phase_shift',1,M^2);
%% �����ź�
tb_length = ceil(T*fs/(M*M));      %% ����Ԫ��Ӧ�Ĳ�������Ŀ
PW_length = round(T*fs);                   %% �����ȶ�Ӧ����Ԫ
P1_signal = zeros(1,PW_length);
for kk = 1:PW_length
    Phi(kk)= phase_shift_array(ceil(kk/tb_length));
    P1_signal(kk) = exp(1i*(2*pi*fc*kk/fs + Phi(kk)+pi/4));   %% �������������ϸ�����λ�������ȡֵʱ�����ǻ���-pi~+pi֮�������𵴣���˼���pi/4
 
end
% y=awgn(P1_signal,SNR);
y=P1_signal;
end