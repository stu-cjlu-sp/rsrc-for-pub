function   y=T1(fc,t,n,k,t1,T,SNR)
pha=[];
for j=0:k-1
    pha1=mod(2*pi/n*ceil((k*t1-j*T)*(j*n/T)),2*pi);
    pha=[pha,pha1];
end
if k == 5
    pha=[pha,0];
end
y=exp(1i*(2*pi*fc*t+pha));
% y=awgn(y,SNR);
end
