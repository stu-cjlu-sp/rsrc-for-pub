 addpath(genpath('\tftb-0.2'));              
 addpath 'waveform-types'                    
%% initial parameters configurations 
fs = 100e6;                                  
A = 1;                                       
waveforms = {'BPSK','Costas','Frank','LFM','P1','P2','P3','P4','T1','T2','T3','T4'}; 
fps = 100;                                  

% filter configuration 
g = kaiser(63,0.5);                          
h = kaiser(63,0.5);  
imgSize = 256;                               
%% initial channels 

SNR = -16:2:10;                              
    
for n = 1 : length(SNR)
    
    disp(['SNR = ',sprintf('%+02d',SNR(n))])                       
    datasetCWD = ['test_dataset_different_parameters1\',num2str(SNR(n)),'db'];
    for i = 1 : length(waveforms)
    % create the folders for dataset storage 
     mkdir(fullfile(datasetCWD,waveforms{i}));                   
    end

    for k = 1 : length(waveforms)
        waveform = waveforms{k};
        switch waveform                                                  
            case 'BPSK'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [11,13];
                fc = linspace(fs/6,fs/5,fps);
                fc  =fc(randperm(fps));
                Ncc = 20:24;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    Bar = Lc(randi(2));
                    if Bar == 11
                        phaseCode = [0 0 0 1 1 1 0 1 1 0 1]*pi;
                    elseif Bar == 13
                        phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
                    end
                    wav = type_BPSK(Ncc, fs, A, fc(idx), phaseCode);
                    wav = awgn(wav,SNR(n),'measured');

                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                end

            case 'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [2, 3, 4, 5];
                fcmin = linspace(fs/30,fs/24,fps);
                fcmin = fcmin(randperm(fps));
                N = linspace(512,1920,fps);
                N = round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    NumHop = randperm(Lc(randi(4)));
                    wav = type_Costas(N(idx), fs, A, fcmin(idx), NumHop);
                    wav = awgn(wav,SNR(n),'measured');

                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                end

            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [4,5,6];
                M = [6,7,8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');

                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                end      
            case 'LFM'  
                disp(['Generating ',waveform, ' waveform ...']);  
                fc = linspace(fs/6,fs/5,fps);                     
                fc = fc(randperm(fps));                           
                B = linspace(fs/16, fs/8, fps);                   
                B = B(randperm(fps));                             
                N = linspace(512,1920,fps);                        
                N = round(N(randperm(fps)));                       
                sweepDirections = {'Up','Down'};                  
                waveformfolderCWD = fullfile(datasetCWD,waveform); 
 
                for idx = 1:fps 

                    wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)}); 
                    wav = awgn(wav,SNR(n),'measured');

                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));
 
                end

            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [4,5,6];
                M = [6, 7, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps

                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                end
            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [4,5,6];
                M = [6, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps

                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                end
            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [4,5,6];
                p = [36, 49, 64];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps

                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [4,5,6];
                p = [36, 49, 64];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps

                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                end
            case 'T1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [3,4,5];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                Nps = 2;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps

                    wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));


                end
                
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [3,4,5];
                Nps = 2;
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps

                    wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));
 

                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [3,4,5];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps

                    wav = type_T3(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));

                end
                
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [3,4,5];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps

                    wav = type_T4(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD1,~,~] = FTCWD(wav',t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD1,fullfile(waveformfolderCWD,sprintf('%d.png',idx)));
  
                end
                
            otherwise
                disp('Done!')
        end
        
    end
    
end