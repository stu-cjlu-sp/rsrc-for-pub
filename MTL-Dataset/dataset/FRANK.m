function   y=FRANK(M,fc,T,fs,SNR)
phase_shift=zeros(M,M);     %���������
for jj=1:M
    for ii=1:M
        phase_shift(jj,ii) = 2*pi*(jj-1)*(ii-1)/M;
    end
end
phase_shift=reshape(phase_shift',1,M*M);       %ת����������ǰ��Ҫת��
n1=round(T*fs);                                %һ������Ĳ�������
n2=ceil(n1/(M*M));                             %һ����Ԫ�Ĳ������� ceil�����Ǵ���X����С����
y1=[];
for ii=1:n1                %n1һ������Ĳ�������
    pha(ii)=phase_shift(ceil(ii/n2));       %ceilȡ��
    y1(ii)=exp(1i*(2*pi*fc*ii/fs+pha(ii)));
end
% y=awgn(y1,SNR);
y=y1;
end
