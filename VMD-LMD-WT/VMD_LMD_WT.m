function wav_denoised = VMD_LMD_WT(wav,wav_noisy,K,N,threshold1,threshold2)
% VMD_LMD_WT
% A multilayer decomposition denoising method based on the combination of 
% variational mode decomposition,local mean decomposition,and wavelet thresholding.
% Input and Parameters:
% ---------------------
% wav           - the original signal
% wav_noisy     - the noisy signal to be decomposed
% K             - the number of modes
% N             - the number of sampling points
% threshold1    - determine the noise-dominated IMFs and signal-dominated IMFs
% threshold2    - retain PFs with high correlation coefficients and discard the remaining PFs
%
% Output:
% ---------------------
% wav_denoised  - the denoised signal
% ---------------------

alpha = 6000;       % moderate bandwidth constraint
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 1e-7;         % tolerance of convergence criterion
[imf_I, ~, ~] = VMD(real(wav_noisy), alpha, tau, K, DC, init, tol);
[imf_Q, ~, ~] = VMD(imag(wav_noisy), alpha, tau, K, DC, init, tol);

imf_I=imf_I';
imf_Q=imf_Q';
wav_I=real(wav_noisy);
wav_Q=imag(wav_noisy);
wav_denoised1=zeros(N,1);
wav_denoised2=zeros(N,1);

% denoising of I-channel signals
for i=1:K
    % calculate the correlation coefficient
    CC_I1(1,i)=sum((wav_I-mean(wav_I)).*(imf_I(:,i)-mean(imf_I(:,i))),1)./sqrt(sum(((wav_I-mean(wav_I)).^2),1).*sum(((imf_I(:,i)-mean(imf_I(:,i))).^2),1));
    if CC_I1(1,i)>threshold1
        wav_denoised1=wav_denoised1+imf_I(:,i);
    else
        [PF_I]=LMD(imf_I(:,i)');
        PF_I=PF_I';
        k=min(size(PF_I));
        for j=1:k
            % calculate the correlation coefficient
            CC_I2(1,j)=sum((wav_I-mean(wav_I)).*(PF_I(:,j)-mean(PF_I(:,j))),1)./sqrt(sum(((wav_I-mean(wav_I)).^2),1).*sum(((PF_I(:,j)-mean(PF_I(:,j))).^2),1));
            if CC_I2(1,j)>threshold2
                PF_I(:,j)=wden(PF_I(:,j),'rigrsure','s','sln',5,'sym7');% wavelet Thresholding
                wav_denoised1=wav_denoised1+0.5*PF_I(:,j);
            end
        end
    end
end

% denoising of Q-channel signals
for i=1:K
    % calculate the correlation coefficient
    CC_Q1(1,i)=sum((wav_Q-mean(wav_Q)).*(imf_Q(:,i)-mean(imf_Q(:,i))),1)./sqrt(sum(((wav_Q-mean(wav_Q)).^2),1).*sum(((imf_Q(:,i)-mean(imf_Q(:,i))).^2),1));
    if CC_Q1(1,i)>threshold1
        wav_denoised2=wav_denoised2+imf_Q(:,i);
    else
        x=imf_Q(:,i)';
        [PF_Q] =  LMD(x);
        PF_Q=PF_Q';
        k=min(size(PF_Q));
        for j=1:k
            % calculate the correlation coefficient
            CC_Q2(1,j)=sum((wav_Q-mean(wav_Q)).*(PF_Q(:,j)-mean(PF_Q(:,j))),1)./sqrt(sum(((wav_Q-mean(wav_Q)).^2),1).*sum(((PF_Q(:,j)-mean(PF_Q(:,j))).^2),1));
            if CC_Q2(1,j)>threshold2
                PF_Q(:,j)=wden(PF_Q(:,j),'rigrsure','s','sln',5,'sym7');% wavelet Thresholding
                wav_denoised2=wav_denoised2+0.5*PF_Q(:,j);
            end
        end
    end
end

wav_denoised=wav_denoised1+1i*wav_denoised2;
end

