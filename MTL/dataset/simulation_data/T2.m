function   y=T2(fc,t,T,n,k,t1,SNR)

pha=[];
for j=0:k-1
    pha1=mod(2*pi/n*ceil((k*t1-j*T)*((2*j-k+1)/T)*(n/2)),2*pi);
    pha=[pha,pha1];
end
if k==5
    pha=[pha,0];
end
y=exp(1i*(2*pi*fc*t+pha));
% y=awgn(y,SNR);
end