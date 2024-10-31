clc;clear;

%%
waveform = 'nosignal';
fs = 100e6;
A = 200;
fps = 5;
g=kaiser(63,0.5);
h=kaiser(63,0.5);
imgSize =224;

SNR = -14:2:10;

for n = 1 : length(SNR)

    disp(['SNR = ',sprintf('%+02d',SNR(n))])
    dataset = ['test\test_3_0\',num2str(SNR(n)),'db'];
    waveformfolder = fullfile(dataset,waveform);
    mkdir(fullfile(dataset,waveform));

    fc_LFM = linspace(fs/6,fs/5,fps);
    fc_LFM=fc_LFM(randperm(fps));
    B_LFM = linspace(fs/20, fs/16, fps);
    B_LFM = B_LFM(randperm(fps));
    sweepDirections = {'Up','Down'};
    N_LFM = linspace(512,1920,fps);
    N_LFM=round(N_LFM(randperm(fps)));
    for idx = 1:fps
        wav_LFM = LFM(N_LFM(idx),fs,A,fc_LFM(idx),B_LFM(idx),sweepDirections{randi(2)});
        
        

        fujia = zeros(1,length(wav_LFM));
        wav_LFM = [wav_LFM,fujia];
        wav_nosigal = awgn(wav_LFM,SNR(n),'measured');

        t=1:length(wav_nosigal);
        [TFD,~,~] = FTCWD_noise(wav_nosigal,t,1024,g,h,1,0,imgSize);
        
        imwrite(TFD,fullfile(waveformfolder,sprintf('%s-snr%02d-no%03d.png',waveform,n,idx)));



    end
end






