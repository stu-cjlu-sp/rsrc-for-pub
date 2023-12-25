function s = type_LFM(NumberSamples,fs,A,fc,Df,updown) % ok
pw = NumberSamples/fs;
t = (1:NumberSamples)/fs;
phi0 = 2*pi*rand(1) - pi;
if (updown == "up")||(updown == "Up")||(updown == "UP")
    f = fc + Df/pw*t;
elseif (updown == "down")||(updown == "Down")||(updown == "DOWN")
    f = fc - Df/pw*t;
    % elseif (updown == "updown")||(updown == "UpDown")||(updown == "UPDOWN")
    %     f1 = fc + Df/pw*t(1:NumberSamples/2);
    %     f2 = flip(f1);
    %     f = [f1, f2];
    % elseif (updown == "downup")||(updown == "DownUp")||(updown == "DOWNUP")
    %     f1 = fc - Df/pw*t(1:NumberSamples/2);
    %     f2 = flip(f1);
    %     f = [f1, f2];
else
    disp('Defaut up direction!')
    f = fc + Df/pw*t;
end

s = A*exp(1j*2*pi*f.*t+phi0);
end