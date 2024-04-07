function s = LFM(NumberSamples,fs,A,fc,Df,updown) 
pw = NumberSamples/fs;
t = (1:NumberSamples)/fs;
phi0 = 2*pi*rand(1) - pi;
if (updown == "up")||(updown == "Up")||(updown == "UP")
    f = fc + Df/pw*t;
elseif (updown == "down")||(updown == "Down")||(updown == "DOWN")
    f = fc - Df/pw*t;
else
    disp('Defaut up direction!')
    f = fc + Df/pw*t;
end

s = A*exp(1j*2*pi*f.*t+phi0);
end