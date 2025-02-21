% time:2025-02-21-15:43
%
% source code for "Transferring Common Model Parameters of Chirp-VEP to
% SSVEP-based BCIs for Reducing Calibration Effort"
%
% add data path before runing

%%
clc;
clear all;
close all;
warning off;

% Subjects info 
subjects = {'S1',   'S2',  'S3',  'S4',...
            'S5',   'S6',  'S7',  'S8',...
            'S9',  'S10', 'S11', 'S12',...
            'S13', 'S14', 'S15', 'S16'};
nSubjects = length(subjects);

% Stimulation parameters
nTargets = 40;
nTestBlocks = 7;

% EEG parameters
channels = [1 4 5 6 7 8 13 14 16];  
Oz = 6;
nChannels = length(channels);
Fs = 256;
Chirp_delay = round(0.115 * Fs);      % visual latency chirp-vep
SSVEP_delay = round(0.135 * Fs);      % visual latency ssvep

% chirp parameters
f0 = 8.0;
f1 = 15.0;
p0 = 0*pi;
c  = 1.75;

% FBCCA parameters
nSubBands = 5;
harmonics = 5;
fbCoefs = (1:nSubBands).^(-1.25) + 0.25;

% JFPM
Freqs = [8.0,   9.0,    10.0,   11.0,   12.0,   13.0,   14.0,   15.0,...
         8.2,   9.2,    10.2,   11.2,   12.2,   13.2,   14.2,   15.2,...
         8.4,   9.4,    10.4,   11.4,   12.4,   13.4,   14.4,   15.4,...
         8.6,   9.6,    10.6,   11.6,   12.6,   13.6,   14.6,   15.6,...
         8.8,   9.8,    10.8,   11.8,   12.8,   13.8,   14.8,   15.8];

Phases = [0.00,  1.75,   1.50,   1.25,   1.00,   0.75,   0.50,   0.25,...
          0.35,  0.10,   1.85,   1.60,   1.35,   1.10,   0.85,   0.60,...
          0.70,  0.45,   0.20,   1.95,   1.70,   1.45,   1.20,   0.95,...
          1.05,  0.80,   0.55,   0.30,   0.05,   1.80,   1.55,   1.30,...
          1.40,  1.15,   0.90,   0.65,   0.40,   0.15,   1.90,   1.65];
Phases = Phases * pi;

% reference_signals
reference = cell(1, nTargets);
for tar = 1:nTargets
    reference{tar} = refsig(Freqs(tar), Fs, 2 * Fs, harmonics);
end

% bandpass filter
order = 6;
low = 7;
high = 90;
[bpB, bpA] = butter(order , [low/(Fs/2) , high/(Fs/2)]);

% notch filter 50 Hz
Fo = 50;
BW = 1;
Fo = (Fo/(Fs/2));
BW = (BW/(Fs/2));
[notchB, notchA] = iirnotch(Fo, BW);

% Regularization
lambda = 0.2;
erp_len = round(1 / 8 * Fs);
norm_method = 'Tikhonov';

% Switch case to handle different algorithms
disp('Algorithm: CTSS  1 Source');
nSource_stimuli = 1;
nTarget_stimuli = 20;
Target_stimuli = sort(randperm(40, nTarget_stimuli));
tw = 4;
train_len = tw * Fs;

for sub_i = 1:length(subjects)
    fprintf('%s\n',subjects{sub_i});
    % ================================ %
    %           train CTSS             %
    % ================================ %
    % set data file path
    blocks = 1;
    Filename = ['..\data\',  'Chirp\S', num2str(sub_i), '\B', num2str(blocks), '_raw.mat'];
    % Load data and process
    load(Filename);
    eeg = squeeze(mean(raw_eeg(:, :, 1 + Chirp_delay : train_len + Chirp_delay), 1));
    
    % Filter
    flt_eeg = zeros(nChannels, train_len);  % the time delay has add in the data preprocessing
    for i = 1:nChannels
        tmp = eeg(i, 1 : train_len);
        tmp = filtfilt(bpB, bpA, tmp);           % band pass
        tmp = filtfilt(notchB, notchA, tmp);     % notch 1 times
        tmp = filtfilt(notchB, notchA, tmp);     % notch 2 times
        tmp = filtfilt(notchB, notchA, tmp);     % notch 3 times
        flt_eeg(i, :) = tmp;
    end
    
    ssvep = flt_eeg;

    % calculate original chirp impulse
    [H, my_h, my_hs] = conv_chirp_H(f0, p0, c, Fs, tw);

    % calculate common chirp impulse
    [common_H] = common_chirp_H(H, f0, f1, c, Fs);

    % filter bank analysis
    common_ir = zeros(nSubBands, erp_len);
    common_sf = zeros(nSubBands, nChannels);
    for fb_i = 1:nSubBands
        ssvep_fb = filterbank(ssvep, Fs, fb_i);

        % Decompose an SSVEP corresponding to the i-th stimulus
        y0 = ssvep_fb;
        y0_oz = y0(Oz, :);
        H0 = common_H(:, 1: train_len);
  
        % Sparse Regularization Matrix
        M_y0 = regmat(nChannels, norm_method) * lambda * Fs;
        M_H0 = regmat(erp_len, norm_method) * lambda * Fs;

        % Alternating Least Square
        x_hat_oz = y0_oz * H0' * pinv(H0 * H0' + M_H0);
        w0_old = randn(1, nChannels);
        x_hat_old = w0_old * y0 * H0' * pinv(H0 * H0' + M_H0);
        e_old = norm(w0_old * y0 - x_hat_old * H0);
        my_err = 100;
        iter = 1;
        while (my_err(iter) > 0.001 && iter < 200)
            w0_new = x_hat_old * H0 * y0' * pinv(y0 * y0' + M_y0);
            x_hat_new = w0_new * y0 * H0' * pinv(H0 * H0' + M_H0);
            e_new = norm(w0_new * y0 - x_hat_new * H0);

            iter = iter + 1;
            my_err(iter) = abs(e_old - e_new);
            w0_old = w0_new;
            w0_old = w0_old / std(w0_old);
            x_hat_old = x_hat_new;
            x_hat_old = x_hat_old / std(x_hat_old);
            e_old = e_new;
        end
        x_hat = x_hat_new;

        r = corrcoef(x_hat, x_hat_oz);
        if r(1, 2) < 0
            x_hat  = -x_hat;
            w0_new = -w0_new;
        end
        common_ir(fb_i, :) = x_hat;
        common_sf(fb_i, :) = w0_new;
    end
    % ==========================  END Training  ======================== %
    
    % ================================================================== %
    %                            Test Stage                              %
    % ================================================================== %
    for testBlocks = 1:nTestBlocks
        fprintf('subject %d-block %d:\tacurracy =  ',sub_i, testBlocks);
        Filename = ['..\data\',  'SSVEP\S', num2str(sub_i), '\B', num2str(testBlocks), '_raw.mat'];

        % Load test data and analysis
        load(Filename);
        for timeWin = 0.2:0.2:1.6
            sig_len = round(timeWin * Fs);
            % =================== %
            %       CTSS test
            % =================== %
            CLASS = zeros(1,nTarget_stimuli);
            for tar = 1:nTarget_stimuli
                testSSVEP = squeeze(raw_eeg(Target_stimuli(tar), :, 1 + SSVEP_delay : sig_len + SSVEP_delay));

                % Filter
                flt_eeg = zeros(nChannels, sig_len);
                for i = 1:nChannels
                    tmp = testSSVEP(i, :);
                    tmp = filtfilt(bpB, bpA, tmp);           % band pass
                    tmp = filtfilt(notchB, notchA, tmp);     % notch 1 times
                    tmp = filtfilt(notchB, notchA, tmp);     % notch 2 times
                    tmp = filtfilt(notchB, notchA, tmp);     % notch 3 times
                    flt_eeg(i, :) = tmp;
                end

                ssvep = flt_eeg;
                rho = zeros(nSubBands, nTarget_stimuli);
                for fb_i = 1:nSubBands
                    ssvep_fb = filterbank(ssvep, Fs, fb_i);
                    for i = 1:nTarget_stimuli
                        f_i = Freqs(Target_stimuli(i));
                        p_i = Phases(Target_stimuli(i));
                        freq_period = 1.05 * (1 / f_i);
                        [H1, H1_t] = conv_H_trans(f_i, p_i, Fs, timeWin, 60, freq_period);

                        % Constructed templates
                        conv_signal = common_ir(fb_i, :) * H1_t;

                        % Spatial filtered templates
                        spt_signal  = common_sf(fb_i, :) * ssvep_fb;

                        % Correlation analysis
                        tmp_r = corrcoef(conv_signal, spt_signal);
                        r1 = tmp_r(1, 2);

                        % CCA analysis
                        [~, ~, tmp_r] = cca(spt_signal, reference{Target_stimuli(i)}(:, 1:sig_len));
                        r2 = tmp_r(1, 1);
    
                        % Feature vectors
                        coefficience = [r1,r2];
                        r_comb = sum(sign(coefficience).*coefficience.^2);
                        rho(fb_i, i) = r_comb;
                    end
                end
                rho_tar = fbCoefs * rho;
                [~, class] = max(rho_tar);
                CLASS(tar) = Target_stimuli(class);
            end      
            % Print Acc
            fprintf('%.2f%%\t',sum(CLASS == Target_stimuli)/nTarget_stimuli*100);
        end
        fprintf('\n');
    end
end