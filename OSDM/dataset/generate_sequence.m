clear; clc; close all;
%%
snr_values = 10:-2:-16;

generation_path = 'C:\Users\';
data4 = load('C:\Users\');
data_original = data4.data;
label_cate_original = data4.label_cate;

for k = 1:length(snr_values)
    current_snr = snr_values(k);
    num_samples = 6000; 
    data = zeros(num_samples, 1000, 2);
    label_box = zeros(num_samples, 4, 2);
    label_cate = zeros(num_samples, 4);

    for i = 1:num_samples
        num_signals = randi([2, 4]);
        indices = randperm(size(data_original, 1), num_signals);
        combined_signal = [];
        start_idx = 1;
        for j = 1:num_signals
            signal = data_original(indices(j), :, :);
            signal = reshape(signal, [], 2);
            remaining_length = 1000 - size(combined_signal, 1);
            if size(signal, 1) > remaining_length
                continue;
            end

            if j == 1
                max_front_padding = min(200, remaining_length - size(signal, 1));
                front_padding = randi([0, max_front_padding]);
                combined_signal = [zeros(front_padding, 2); signal];
                start_idx = front_padding + 1;
            else
                max_inter_padding = max(0, remaining_length - size(signal, 1));
                inter_padding = randi([0, max_inter_padding]);
                combined_signal = [combined_signal; zeros(inter_padding, 2); signal];
            end
            end_idx = start_idx + size(signal, 1) - 1;
            label_box(i, j, 1) = start_idx;
            label_box(i, j, 2) = end_idx;
            label_cate(i, j) = label_cate_original(indices(j));
            start_idx = end_idx + 1;
        end

        if size(combined_signal, 1) < 1000
            combined_signal = [combined_signal; zeros(1000 - size(combined_signal, 1), 2)];
        end

        data(i, :, :) = combined_signal;
    end

    for i = 1:size(data, 1)
        for dim = 1:size(data, 3)
            data(i, :, dim) = awgn(data(i, :, dim), current_snr, 'measured');
        end
    end
    
    snr = current_snr * ones(size(data, 1), 1);
    
    save(fullfile(generation_path, ['data_' num2str(k) '.mat']), 'data','snr');
    save(fullfile(generation_path, ['box_' num2str(k) '.mat']), 'label_box');
    save(fullfile(generation_path, ['cate_' num2str(k) '.mat']), 'label_cate');
    
end

%%
data = [];
label_box = [];
label_cate = [];
snr = [];

for k = 1:length(snr_values)
    data_file_name = fullfile(generation_path, ['data_' num2str(k) '.mat']);
    box_file_name = fullfile(generation_path, ['box_' num2str(k) '.mat']);
    cate_file_name = fullfile(generation_path, ['cate_' num2str(k) '.mat']);
    data_all = load(data_file_name);
    data_original = data_all.data;
    snr_original = data_all.snr;

    box_data = load(box_file_name);
    label_box_original = box_data.label_box;

    cate_data = load(cate_file_name);
    label_cate_original = cate_data.label_cate;

    assert(size(data_original, 1) == 6000 && size(data_original, 2) == 1000 && size(data_original, 3) == 2, '信号形状不符合要求');
    assert(size(label_box_original, 1) == 6000 && size(label_box_original, 2) == 4 && size(label_box_original, 3) == 2, 'label_box 形状不符合要求');
    assert(size(label_cate_original, 1) == 6000 && size(label_cate_original, 2) == 4, 'label_cate 形状不符合要求');

    data = cat(1, data, data_original);
    snr = cat(1, snr, snr_original);
    label_box = cat(1, label_box, label_box_original);
    label_cate = cat(1, label_cate, label_cate_original);
end

final_save_path = 'C:\Users';
save(fullfile(final_save_path, 'signal.mat'), 'data','snr');
save(fullfile(final_save_path, 'label_box.mat'), 'label_box');
save(fullfile(final_save_path, 'label_cate.mat'), 'label_cate');