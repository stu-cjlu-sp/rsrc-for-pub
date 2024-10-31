clc;clear;

%%
fs = 100e6;
A = 1;
waveforms = {'LFM','BPSK','Costas','Frank'};

waveforms_two ={};
for i = 1 : length(waveforms)
    for ii =(i+1):length(waveforms)
        waveforms_comb = [waveforms{i},'+',waveforms{ii}];
        waveforms_two(end+1) = {waveforms_comb};
    end
end

fps = 200;
g=kaiser(63,0.5);
h=kaiser(63,0.5);
imgSize =224;

SNR = -14:2:10;

for n = 1 : length(SNR)

    disp(['SNR = ',sprintf('%+02d',SNR(n))])
    dataset = ['test\test_3_2\',num2str(SNR(n)),'db'];

    for i = 1 : length(waveforms_two)
        mkdir(fullfile(dataset,waveforms_two{i}));
    end
    
    %LFM
    fc_LFM = linspace(fs/6,fs/5,fps);
    fc_LFM=fc_LFM(randperm(fps));
    B_LFM = linspace(fs/20, fs/16, fps);
    B_LFM = B_LFM(randperm(fps));
    sweepDirections = {'Up','Down'};
    N_LFM = linspace(512,1920,fps);
    N_LFM=round(N_LFM(randperm(fps)));
    %Costas
    Lc_Costas = [4,4,4];
    fcmin_Costas = linspace(fs/15,fs/10,fps);
    fcmin_Costas=fcmin_Costas(randperm(fps));
    N_Costas = linspace(512,1920,fps);
    N_Costas=round(N_Costas(randperm(fps)));
    %Frank
    fc_Frank = linspace(fs/6,fs/5,fps);
    fc_Frank=fc_Frank(randperm(fps));
    Ncc_Frank = [3,4,5];
    M_Frank = [6, 7, 8];
    %BPSK
    Lc_BPSK = [7,11,13];
    fc_BPSK = linspace(fs/6,fs/5,fps);
    fc_BPSK=fc_BPSK(randperm(fps));
    Ncc_BPSK = 20:24;

    for idx = 1:fps
        %LFM
        wav_LFM = LFM(N_LFM(idx),fs,A,fc_LFM(idx),B_LFM(idx),sweepDirections{randi(2)});
        %Costas
        NumHop = randperm(Lc_Costas(randi(3)));
        wav_Costas = Costas(N_Costas(idx), fs, A, fcmin_Costas(idx), NumHop);
        %Frank
        wav_Frank = Frank(Ncc_Frank(randi(3)), fs, A, fc_Frank(idx), M_Frank(randi(3)));
        %BPSK
        Bar = Lc_BPSK(randi(3));
        if Bar == 7
            phaseCode = [0 0 0 1 1 0 1]*pi;
        elseif Bar == 11
            phaseCode = [0 0 0 1 1 1 0 1 1 0 1]*pi;
        elseif Bar == 13
            phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
        end
        wav_BPSK = BPSK(Ncc_BPSK, fs, A, fc_BPSK(idx), phaseCode);

        for i = 1 : length(waveforms)
            for ii =(i+1):length(waveforms)
                waveforms_comb = [waveforms{i},'+',waveforms{ii}];
                waveformfolder = fullfile(dataset,waveforms_comb);

                wav1 = ['wav_',waveforms{i}];
                wav1 = eval(wav1);
                wav1=preprocessing(wav1);
                wav2 = ['wav_',waveforms{ii}];
                wav2 = eval(wav2);
                wav2=preprocessing(wav2);

                original_length = length(wav2);
                desired_length = length(wav1);
                x_original = 1:original_length;
                x_desired = linspace(1, original_length, desired_length);
                wav2 = interp1(x_original, wav2, x_desired);

                wav = wav1+wav2;
                wav = awgn(wav,SNR(n),'measured');
                t=1:length(wav);

                [TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                imwrite(TFD,fullfile(waveformfolder,sprintf('%s-snr%02d-no%03d.png',waveforms_comb,n,idx)));

            end
        end

    end
end






