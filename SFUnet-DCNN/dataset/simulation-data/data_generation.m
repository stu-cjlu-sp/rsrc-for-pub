 addpath(genpath('\tftb-0.2'));              
 addpath 'waveform-types'                    
%% initial parameters configurations 
fs = 100e6;                                  % sample frequency 
A = 1;                                       % amplitude 
waveforms = {'BPSK','Costas','Frank','LFM','P1','P2','P3','P4','T1','T2','T3','T4'}; 
fps = 200;                                  % the number of signal per SNR per waveform codes 

% filter configuration 
g = kaiser(63,0.5);                          
h = kaiser(63,0.5);  
imgSize = 256;                               
%% initial channels 

SNR = -16:2:10 ;                               % snr range 
    
for n = 1 : length(SNR)
    
    disp(['SNR = ',sprintf('%+02d',SNR(n))])                  
    datasetCWD = ['test_dataset_denoising4\',num2str(SNR(n)),'db'];
    datasetCWD1 = ['test_noise_dataset_denoising4\',num2str(SNR(n)),'db'];

    for i = 1 : length(waveforms)
    % create the folders for dataset storage 
     mkdir(fullfile(datasetCWD,waveforms{i}));                     
     mkdir(fullfile(datasetCWD1,waveforms{i}));                    
    end

    for k = 1 : length(waveforms)
        waveform = waveforms{k};
        switch waveform                                               
            case 'BPSK'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = 20:24;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps
                    Bar = Lc(randi(3));
                    if Bar == 7
                        phaseCode = [0 0 0 1 1 0 1]*pi;
                    elseif Bar == 11
                        phaseCode = [0 0 0 1 1 1 0 1 1 0 1]*pi;
                    elseif Bar == 13
                        phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
                    end
                    wav = type_BPSK(Ncc, fs, A, fc(idx), phaseCode);
                    
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));

                end

            case 'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [3, 4, 5, 6];
                fcmin = linspace(fs/30,fs/24,fps);
                fcmin = fcmin(randperm(fps));
                N = linspace(512,1920,fps);
                N = round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps
                    NumHop = randperm(Lc(randi(4)));
                    wav = type_Costas(N(idx), fs, A, fcmin(idx), NumHop);

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end

            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6,7,8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end      
            case 'LFM'  
                disp(['Generating ',waveform, ' waveform ...']);   
                fc = linspace(fs/6,fs/5,fps);                      
                fc = fc(randperm(fps));                            
                B = linspace(fs/20, fs/16, fps);                   
                B = B(randperm(fps));                              
                N = linspace(512,1920,fps);                        
                N = round(N(randperm(fps)));                       
                sweepDirections = {'Up','Down'};                   
                waveformfolderCWD = fullfile(datasetCWD,waveform); 
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform); 
 
                for idx = 1:fps 

                    wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)}); 

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end

            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc = fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps

                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end
            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps

                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end

            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps

                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));

                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps

                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end
            case 'T1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [4,5,6];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                Nps = 2;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps

                    wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end
                
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [4,5,6];
                Nps = 2;
                N = linspace(512,1920,fps);
                N = round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps

                    wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc = fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [4,5,6];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps

                    wav = type_T3(N(idx), fs, A, fc(idx), Nps,B(idx));

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end
                
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc = fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [4,5,6];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                waveformfolderCWD1 = fullfile(datasetCWD1,waveform);
                for idx = 1:fps

                    wav = type_T4(N(idx), fs, A, fc(idx), Nps,B(idx));

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));
                    
                    wav1 = awgn(wav,SNR(n),'measured');
                    [CWD_TFD1,~,~] = FTCWD(wav1',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD1,sprintf('%d.png',idx)));
                end
                
            otherwise
                disp('Done!')
        end
        
    end
    
end