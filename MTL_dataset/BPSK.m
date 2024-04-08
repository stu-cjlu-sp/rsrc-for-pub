function   y=BPSK(fc,t,n_bpsk,code_bpsk,N_bpsk,SNR)
y=[];
for i=1:n_bpsk
    if code_bpsk(i)==0
        pha=pi;
    else
        pha=0;
    end
    y1=exp(1i*(2*pi*fc*t((i-1)*N_bpsk+1:i*N_bpsk)+pha));
    y=[y,y1];
end 
