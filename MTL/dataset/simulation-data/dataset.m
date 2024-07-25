clc
clear
close all
N=256;
nfft=256;
fsz=23;
winlen=32;
%% SIMU1
num=200;
time_domain=zeros(num*12*11,256);
MSST_label=int8(zeros(num*12*11,256,256));

for SNR=0:2:20
    for j=1:1:num
        M=randi([4,8],1,1);
        fs=200e6;
        fc= randi([0.125*fs,0.25*fs],1,1);
        T=1.28e-6;
        t=0:1/fs:T-1/fs;
        B=randi([0.0625*fs,0.125*fs],1,1);
        K=B/T;
        n=2;
        k=randi([4,5],1,1);
        t1=0:1/fs:(T/k)-1/fs;
        w=randi([1,3],1,1);
        switch (w)
          case 1
              code_bpsk=[1 1 1 0 0 1 0];
          case 2
              code_bpsk=[1 1 1 0 0 0 1 0 0 1 0];
          case 3
              code_bpsk=[1 1 1 1 1 0 0 1 1 0 1 0 1];
        end
        n_bpsk=length(code_bpsk);
        pw_bpsk=T/n_bpsk;
        t_bpsk=0:1/fs:pw_bpsk-1/fs;
        N_bpsk=length(t_bpsk);
        f1=randi([(1/80)*fs,(1/2)*fs],1,1);
        f2=randi([(1/80)*fs,(1/2)*fs],1,1);
        f3=randi([(1/80)*fs,(1/2)*fs],1,1);
        f4=randi([(1/80)*fs,(1/2)*fs],1,1);
        code_4fsk=[0 0 1 0 0 1 1 1];
        n_4fsk=length(code_4fsk)/2;
        pw_4fsk=T/n_4fsk;
        t_4fsk=0:1/fs:pw_4fsk-1/fs;
        N_4fsk=length(t_4fsk);
        F=16e6;

        %%
        y=FSK4(f1,f2,f3,f4,t,n_4fsk,code_4fsk,N_4fsk,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+0*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        noisy_y=Y/max2;
        time_domain(j+SNR*num/2+0*num*11,:)=noisy_y;


        y=BPSK(fc,t,n_bpsk,code_bpsk,N_bpsk,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR/2*num+1*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        [~,Q]=size(Y);
        time_domain(j+SNR/2*num+1*num*11,1:Q)=Y;

        y=LFM(fc,t,K,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+2*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+2*num*11,:)=Y;

        y=FRANK(M,fc,T,fs,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+3*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+3*num*11,:)=Y;

        y=P1(M,fc,T,fs,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+4*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+4*num*11,:)=Y;

        y=P2(M,fc,T,fs,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+5*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+5*num*11,:)=Y;


        y=P3(M,fc,T,fs,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+6*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+6*num*11,:)=Y;

        y=P4(M,fc,T,fs,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+7*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+7*num*11,:)=Y;

        y=T1(fc,t,n,k,t1,T,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+8*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+8*num*11,:)=Y;

        y=T2(fc,t,T,n,k,t1,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+9*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+9*num*11,:)=Y;

        y=T3(fc,t,n,F,T,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+10*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+10*num*11,:)=Y;

        y=T4(fc,t,n,T,F,SNR);
        TFD=MSST_Y(y);
        max1=max(max(TFD));
        TFD=abs(TFD)/max1;
        TFD=int8(TFD*255);
        MSST_label(j+SNR*num/2+11*num*11,:,:)=TFD;
        Y=awgn(y,SNR);
        max2=max(Y);
        Y=Y/max2;
        time_domain(j+SNR*num/2+11*num*11,:)=Y;
    end
end