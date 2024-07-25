function   y=FSK4(f1,f2,f3,f4,t,n_4fsk,code_4fsk,N_4fsk,SNR)
y=[];
for i=1:n_4fsk
    if code_4fsk(2*i-1)==0&&code_4fsk(2*i)==0
        fc=f1;
    elseif code_4fsk(2*i-1)==0&&code_4fsk(2*i)==1
        fc=f2;
    elseif code_4fsk(2*i-1)==1&&code_4fsk(2*i)==1
        fc=f3;
    else
        fc=f4;
    end
    y1=exp(1i*(2*pi*fc*t((i-1)*N_4fsk+1:i*N_4fsk)));
    y=[y,y1];
end

% y=awgn(y,SNR);
% plot(t,y)
end