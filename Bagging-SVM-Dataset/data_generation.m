% addpath(genpath("\tftb-0.2")); 

fs = 100e6; 
A = 1;      
waveforms = {'LFM','Costas','BPSK','Frank','P1','P2','P3','P4','T1','T2','T3','T4'};   


fps = 1;  
g=kaiser(63,0.5);
h=kaiser(63,0.5);
imgSize =224;

SNR =-14:2:10;     
    
for n = 1 : length(SNR)
    
    disp(['SNR = ',sprintf('%+02d',SNR(n))])
    datasetCWD = ['CWD\',num2str(SNR(n)),'db'];
    for i = 1 : length(waveforms)

    mkdir(fullfile(datasetCWD,waveforms{i}));
    end

    for k = 1 : length(waveforms)
        waveform = waveforms{k};
        switch waveform

            case 'LFM'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20, fs/16, fps);
                B = B(randperm(fps));
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                sweepDirections = {'Up','Down'};
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    wav = awgn(wav,SNR(n),'measured');           
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('lfm-snr%02d-no%05d.png',n,idx)));
                    
                end
            case 'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [3, 4, 5, 6];
                fcmin = linspace(fs/30,fs/24,fps);
                fcmin=fcmin(randperm(fps));
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    NumHop = randperm(Lc(randi(4)));
                    wav = Costas(N(idx), fs, A, fcmin(idx), NumHop);
                    wav = awgn(wav,SNR(n),'measured');

                    

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('costas-snr%02d-no%05d.png',n,idx)));
                    
                end
                
            case 'BPSK'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
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
                    wav = BPSK(Ncc, fs, A, fc(idx), phaseCode);
                    wav = awgn(wav,SNR(n),'measured');

                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('bpsk-snr%02d-no%05d.png',n,idx)));
                    
                end
            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('frank-snr%02d-no%05d.png',n,idx)));
                    
                end
            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('p1-snr%02d-no%05d.png',n,idx)));

                end
            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('p2-snr%02d-no%05d.png',n,idx)));

                end
            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('p3-snr%02d-no%05d.png',n,idx)));

                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize); 
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('p4-snr%02d-no%05d.png',n,idx)));

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
                for idx = 1:fps
                    wav = T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('t1-snr%02d-no%05d.png',n,idx)));

                end
                
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [4,5,6];
                Nps = 2;
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('t2-snr%02d-no%05d.png',n,idx)));

                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [4,5,6];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = T3(N(idx), fs, A, fc(idx), Nps, B(idx));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('t3-snr%02d-no%05d.png',n,idx)));

                end
                
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [4,5,6];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    wav = T4(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('t4-snr%02d-no%05d.png',n,idx)));
                    
                end
                
            otherwise
                disp('Done!')
        end
        
    end
    
end
