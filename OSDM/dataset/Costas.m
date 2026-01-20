% % function y = Costas(fc, T, fs, SNR)
% %     % 生成 Costas 序列（典型为长度 M 的排列，例如 13 或 10）
% %     costas_seq = [0 7 2 5 1 8 3 10 4 9 6 11 12];  % 13 元 Costas 序列
% %     M = length(costas_seq);
% %     symbol_time = T / M;
% %     t = 0:1/fs:T-1/fs;
% % 
% %     % 生成每个频率跳变子脉冲
% %     y = [];
% %     for i = 1:M
% %         idx_start = floor((i-1) * symbol_time * fs) + 1;
% %         idx_end = floor(i * symbol_time * fs);
% %         t_segment = (0:(idx_end - idx_start)) / fs;
% %         max_offset = 20e6;  % 最大偏移量20MHz
% %         freq_step = max_offset / (M-1);
% %         freq = (costas_seq(i) - floor(M/2)) * freq_step;
% %         % freq = costas_seq(i) * fs / (2*M);  % 把频率映射到合理范围
% %         y_segment = exp(1j * 2 * pi * (fc + freq) * t_segment);
% %         y = [y, y_segment];
% %     end
% % 
% %     % 修正长度（标准化为和其他信号一致）
% %     y = pad_or_truncate(y, 256);
% % end
% 
% function y = Costas(fc, T, fs, SNR)
% 
%     costas_seq = [0 7 2 5 1 8 3 10 4 9 6 11 12];
%     M = min(length(costas_seq), 13);
% 
%     symbol_time = T / M;
%     total_samples = round(T * fs);
% 
% 
%     t = (0:total_samples-1) / fs;
% 
%     y = zeros(1, total_samples);
%     max_offset = 20e6;
%     freq_step = max_offset / (M-1);
% 
%     for i = 1:M
%         idx_start = floor((i-1) * symbol_time * fs) + 1;
%         idx_end = min(floor(i * symbol_time * fs), total_samples);
% 
%         if idx_start <= idx_end
%             freq_offset = (costas_seq(i) - floor(M/2)) * freq_step;
% 
%             t_segment = t(idx_start:idx_end) - (i-1) * symbol_time;
%             y(idx_start:idx_end) = exp(1j * 2 * pi * (fc + freq_offset) * t_segment);
%         end
%     end
% 
%     % y = awgn(y, SNR);
%     % 
%     % y = y / max(abs(y));
% end




function y = Costas(fh, T, fs, SNR, Nc)  % 1. 新增Nc参数，基频名改为fh
    % 2. 根据Nc选择对应的3/4/5位Costas序列（标准二元Costas序列）
    switch Nc
        case 3
            costas_seq = [0, 1, 3];  % 3位Costas序列
        case 4
            costas_seq = [0, 1, 3, 2];  % 4位Costas序列
        case 5
            costas_seq = [0, 2, 1, 4, 3];  % 5位Costas序列
    end
    symbol_time = T / Nc;  % 3. 用Nc（而非固定M）计算码元时间
    total_samples = round(T * fs);
    t = (0:total_samples-1) / fs;
    y = zeros(1, total_samples);
    max_offset = 20e6;  % 保持原频率偏移范围（其他不变）
    freq_step = max_offset / (Nc - 1);  % 4. 用Nc计算频率步长
    % 5. 用Nc（而非固定M）循环生成每个码元
    for i = 1:Nc
        idx_start = floor((i-1) * symbol_time * fs) + 1;
        idx_end = min(floor(i * symbol_time * fs), total_samples);
        if idx_start <= idx_end
            freq_offset = (costas_seq(i) - floor(Nc/2)) * freq_step;
            t_segment = t(idx_start:idx_end) - (i-1) * symbol_time;
            y(idx_start:idx_end) = exp(1i * 2 * pi * (fh + freq_offset) * t_segment);  % 6. 用fh作为基频
        end
    end
end