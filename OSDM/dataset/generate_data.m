clc
clear
close all
N=256;
nfft=256;
fsz=23;
winlen=32;
%% SIMU1

num=35000;
num_signals = 12;
data=zeros(num*num_signals,256,2);
label_cate = zeros(num*num_signals, 1);
SNR=0;

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
    F=16e6;
    fh = rand(1,1) * (fs/10 - fs/15) + fs/15;
    Nc = randi([3,5],1,1); 

    %% BPSK
    y=BPSK(fc,t,n_bpsk,code_bpsk,N_bpsk,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    [~,Q]=size(Y);
    real_part = real(Y);
    imag_part = imag(Y);
    
    data((j-1)*num_signals + 2, 1:Q, 1) = real_part;
    data((j-1)*num_signals + 2, 1:Q, 2) = imag_part;
    label_cate((j-1)*num_signals + 2) = 1;

    %% Costas
    y = Costas(fh, T, fs, SNR, Nc); 
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 1, :, 1) = real_part;
    data((j-1)*num_signals + 1, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 1) = 2;

    

    %% LFM
    y=LFM(fc,t,K,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 3, :, 1) = real_part;
    data((j-1)*num_signals + 3, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 3) = 3;

    %% FRANK
    y=FRANK(M,fc,T,fs,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 4, :, 1) = real_part;
    data((j-1)*num_signals + 4, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 4) = 4;

    %% P1
    y=P1(M,fc,T,fs,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 5, :, 1) = real_part;
    data((j-1)*num_signals + 5, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 5) = 5;

    %% P2
    y=P2(M,fc,T,fs,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 6, :, 1) = real_part;
    data((j-1)*num_signals + 6, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 6) = 6;

    %% P3
    y=P3(M,fc,T,fs,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 7, :, 1) = real_part;
    data((j-1)*num_signals + 7, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 7) = 7;

    %% P4
    y=P4(M,fc,T,fs,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 8, :, 1) = real_part;
    data((j-1)*num_signals + 8, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 8) = 8;

    %% T1
    y=T1(fc,t,n,k,t1,T,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 9, :, 1) = real_part;
    data((j-1)*num_signals + 9, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 9) = 9;

    %% T2
    y=T2(fc,t,T,n,k,t1,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 10, :, 1) = real_part;
    data((j-1)*num_signals + 10, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 10) = 10;

    %% T3
    y=T3(fc,t,n,F,T,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 11, :, 1) = real_part;
    data((j-1)*num_signals + 11, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 11) = 11;

    %% T4
    y=T4(fc,t,n,T,F,SNR);
    Y=y;
    max2=max(Y);
    Y=Y/max2;
    real_part = real(Y);
    imag_part = imag(Y);
    data((j-1)*num_signals + 12, :, 1) = real_part;
    data((j-1)*num_signals + 12, :, 2) = imag_part;
    label_cate((j-1)*num_signals + 12) = 12;
end

final_save_path = 'C:\Users';
save(fullfile(final_save_path, 'signal.mat'), 'data','label_cate');
