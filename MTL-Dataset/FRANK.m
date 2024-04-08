function   y=FRANK(M,fc,T,fs,SNR)
phase_shift=zeros(M,M);     %创建零矩阵
for jj=1:M
    for ii=1:M
        phase_shift(jj,ii) = 2*pi*(jj-1)*(ii-1)/M;
    end
end
phase_shift=reshape(phase_shift',1,M*M);       %转成列向量，前面要转置
n1=round(T*fs);                                %一个脉冲的采样点数
n2=ceil(n1/(M*M));                             %一个码元的采样点数 ceil函数是大于X的最小整数
y1=[];
for ii=1:n1                %n1一个脉冲的采样点数
    pha(ii)=phase_shift(ceil(ii/n2));       %ceil取整
    y1(ii)=exp(1i*(2*pi*fc*ii/fs+pha(ii)));
end
% y=awgn(y1,SNR);
y=y1;
end
