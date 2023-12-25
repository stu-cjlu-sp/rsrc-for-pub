clear all
addpath(genpath("\tftb-0.2")); % Modify the pathname in your pc
addpath 'waveform-types'

%% initial parameters configurations
fs = 100e6; % sample frequency
A = 1;      % amplitude
waveforms = {'LFM','Costas','BPSK','Frank','P1','P2','P3','P4','T1','T2','T3','T4'};% 12 LPI waveform codes
% datasetCWD = 'E:\EMD\7.4-2\train2';
% for i = 1 : length(waveforms)
%     % create the folders for dataset storage
%     mkdir(fullfile(datasetCWD,waveforms{i}));
% end

fps =400;% the number of signal per SNR per waveform codes
g=kaiser(63,0.5);
h=kaiser(63,0.5);
imgSize = 112;
%%
SNR = -20 : 1 : 10;% snr range

for n = 21
    snr=SNR(n);
    disp(['SNR = ',sprintf('%+02d',SNR(n))])
    
    datasetCWD = ['E:\EMD\7.4-2\test2\',num2str(SNR(n)),'db\'];
    for i = 1 : length(waveforms)
        %create the folders for dataset storage
        mkdir(fullfile(datasetCWD,waveforms{i}));
    end
    
    for K = 1 : length(waveforms)
        waveform = waveforms{K};
        switch waveform
            case 'LFM'
                disp(['Generating ',waveform, ' waveform ...']);
                
                % Define parameters
                fc = linspace(fs/6,fs/5,fps);
                fc = fc(randperm(fps));     % Randomize carrier frequencies
                B = linspace(fs/20, fs/16, fps);
                B = B(randperm(fps));       % Randomize bandwidths
                N = linspace(1024,1920,fps);
                N = round(N(randperm(fps)));% Randomize signal lengths
                sweepDirections = {'Down','Up'};
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate LFM waveform
                    wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    wav = wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2; % modes
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,N(idx),0.2,0.05);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('LFM-snr%02d-no%05d.png',snr,idx)));
                end
                
            case  'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                % Define parameters
                Lc = [3,4,5,6];
                fcmin = linspace(fs/30,fs/24,fps);
                fcmin=fcmin(randperm(fps)); % Randomize carrier frequencies
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));  % Randomize signal lengths
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    NumHop = randperm(Lc(randi(3)));
                    % Generate waveform
                    wav = type_Costas(N(idx), fs, A, fcmin(idx), NumHop);
                    wav = wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2;     % modes;
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,N(idx),0.2,0.05);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('Costas-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'BPSK'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/13,fs/10,fps);
                fc = fc(randperm(fps));
                Ncc = 20:24;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    Bar = Lc(randi(3));
                    if Bar == 7
                        phaseCode = [0 0 0 1 1 0 1]*pi;
                    elseif Bar == 11
                        phaseCode = [0 0 0 1 1 1 0 1 1 0 1]*pi;
                    elseif Bar == 13
                        phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
                    end
                    wav = type_Barker(Ncc, fs, A, fc(idx), phaseCode);
                    L = length(wav);
                    wav = wav';
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2 ;             % modes;
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,L,0.2,0.01);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('BPSK-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc = fc(randperm(fps));     % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                M = [6, 7, 8];              % Number of frequency steps
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    L = length(wav);
                    wav=wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K =  7+(n-11)/2;% modes;
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,L,0.2,0.01);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('Frank-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                Ncc = [3,4,5];          % Cycles per phase code
                M = [6, 7, 8];          % Number of frequency steps
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    L = length(wav);
                    wav=wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2;% modes;
                     % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,L,0.3,0.1);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('P1-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                Ncc = [3,4,5];          % Cycles per phase code
                M = [6, 8];             % Number of frequency steps
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    L = length(wav);
                    wav = wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2;% modes
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,L,0.1,0.02);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('P2-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));       % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                p = [36, 49, 64];           % Number of subcodes
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    L = length(wav);
                    wav =wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2;% modes
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,L,0.3,0.1);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('P3-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));       % Randomize carrier frequencies
                Ncc = [3,4,5];              % Cycles per phase code
                p = [36, 49, 64];           % Number of subcodes
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    L = length(wav);
                    wav = wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2;% modes
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,L,0.3,0.05);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('P4-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'T1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                Ng = [4,5,6];           % Number of segments
                Nps = 2;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    L = length(wav);
                    wav = wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2;% modes
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,L,0.3,0.1);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('T1-snr%02d-no%05d.png',snr,idx)));
                end
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                Ng = [4,5,6];           % Number of segments
                Nps = 2;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                    L = length(wav);
                    wav = wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11)/2;% modes;
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,L,0.3,0.05);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('T2-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/13,fs/10,fps);
                fc=fc(randperm(fps));   % Randomize carrier frequencies
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));   % Randomize bandwidths
                Nps = 2;
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));% Randomize signal lengths
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_T3(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav = wav';
                     % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+3*(n-11)/2;% modes
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,N(idx),0.2,0.1);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                    % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('T3-snr%02d-no%05d.png',snr,idx)));
                end
                
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/13,fs/10,fps);
                fc=fc(randperm(fps));       % Randomize carrier frequencies
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));       % Randomize bandwidths
                Nps = 2;
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));  % Randomize signal lengths
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    % Generate waveform
                    wav = type_T4(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav =wav';
                    % Add white Gaussian noise to the waveform
                    wav_noisy = awgn(wav,SNR(n),'measured');
                    K = 7+(n-11); % modes;
                    % Denoise the waveform using VMD_LMD_WT
                    wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,N(idx),0.2,0.01);
                    t=1:length(wav_denoised);
                    [CWD_yt,~,~] = FTCWD(wav_denoised,t,1024,g,h,1,0,imgSize);
                     % Display and save the CWD image
                    imshow(CWD_yt)
                    imwrite(CWD_yt,fullfile(waveformfolderCWD,sprintf('T4-snr%02d-no%05d.png',snr,idx)));
                end
            otherwise
                disp('Done!')
        end
    end
end