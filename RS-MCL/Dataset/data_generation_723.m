
clear all
addpath(genpath("\tftb-0.2")); % Modify the pathname in your pc
addpath 'waveform-types'

%% initial parameters configurations
fs = 100e6; % sample frequency
A = 1;      % amplitude
waveforms = {'LFM','BPSK','Costas','Frank','P1', 'P3','P2','P4','T1', 'T2','T3','T4'};
datasetCWD1 = 'E:\1\train_IQ';
datasetCWD1 = 'E:\1\train_cwd';
for i = 1 : length(waveforms)
        mkdir(fullfile(datasetCWD1,waveforms{i}));
        mkdir(fullfile(datasetCWD2,waveforms{i}));
end

fps =300;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
g=kaiser(63,0.5);
h=kaiser(63,0.5);
imgSize = 64;
T=1e-6; 
SNR = -20 : 1 : 20;     % snr range                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                
for n = 7:2:31
    snr=SNR(n);
    disp(['SNR = ',sprintf('%+02d',SNR(n))])
%     datasetCWD1 = ['E:\1\test\',num2str(SNR(n)),'db\'];
%     for i = 1 : length(waveforms)
%         mkdir(fullfile(datasetCWD1,waveforms{i}));
%     end

    for k = 1 : 12
        waveform = waveforms{k};
        switch waveform
            case 'LFM'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20, fs/16, fps);
                B = B(randperm(fps));
                N = linspace(1024,1024,fps);
                N=round(N(randperm(fps)));
                sweepDirections = {'Down','Up'};
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                for idx = 1:fps
                    wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    wav = wav';
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,I2,Q2);
%                     figure(1)
                    t=1:length(wav);
                    [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
%                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','LFM-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
                    save(strcat(waveformfolderCWD2,'/','LFM-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end

             case  'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [3,4,5,6];
                fcmin = linspace(fs/30,fs/24,fps);
                fcmin=fcmin(randperm(fps));
                N = linspace(1024,1024,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_2=zeros(1,fps);
                for idx = 1:fps
                    NumHop = randperm(Lc(randi(3)));
                    wav = type_Costas(N(idx), fs, A, fcmin(idx), NumHop);
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,I2,Q2);
%                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
%                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','Costas-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','Costas-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end

            case 'BPSK'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = 20:24;
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_2=zeros(1,fps);
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
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,I2,Q2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','BPSK-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','BPSK-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
                
            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,I2,Q2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav); 
                    save(strcat(waveformfolderCWD1,'/','Frank-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','Frank-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,Q2,I2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','P1-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','P1-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
            case 'P2'
                disp(['Generating ',waveform, 'waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 8];
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,I2,Q2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','P2-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','P2-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,Q2,I2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','P3-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','P3-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,I2,Q2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','P4-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','P4-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
            case 'T1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [4,5,6];
                N = linspace(1024,1024,fps);
                N=round(N(randperm(fps)));
                Nps = 2;
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,I2,Q2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','T1-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','T1-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [4,5,6];
                Nps = 2;
                N = linspace(1024,1024,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,I2,Q2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','T2-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','T2-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [4,5,6];
                Nps = 2;
                N = linspace(1024,1024,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_T3(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,Q2,I2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','T3-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','T3-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [4,5,6];
                Nps = 2;
                N = linspace(1024,1024,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
%                 waveformfolderCWD2 = fullfile(datasetCWD2,waveform);
                snr_1=zeros(1,fps);
                for idx = 1:fps
                    wav = type_T4(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav = wav';
                    t=1:length(wav);
                    y = awgn(wav,SNR(n),'measured');                    
                    I2=real(y);
                    Q2=imag(y);
                    y_output= cat(2,Q2,I2);
% %                     figure(1)
%                     t=1:length(wav);
%                     [CWD_wav,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
% %                     imshow(CWD_wav);
                    save(strcat(waveformfolderCWD1,'/','T4-snr',num2str(snr),'-no',num2str(idx),'.mat'),'y_output');
%                     save(strcat(waveformfolderCWD2,'/','T4-snr',num2str(snr),'-no',num2str(idx),'.mat'),'CWD_wav');
                end
            otherwise
                disp('Done!')
        end
        
    end
    
end