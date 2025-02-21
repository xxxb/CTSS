function [H, my_h, my_hs] = conv_chirp_H(f0, p, c, Fs, tw) % f0, p, c, Fs, tw

% f0:                initial frequency
% p:                initial phase
% c:                chirp rate
% Fs:               sampling rate
% r:                impulse response

sig_len = round(tw * Fs);
t = (0 : sig_len-1) / Fs;
h0 = cos(2 * pi * f0 .* t + pi * c .* t .* t + p) + 1;
sel_idx = round(1 * (0 : Fs*tw-1)) + 1;
h_val = h0(1);
cn = 1;

h = zeros(1, sig_len);  % Initialization chirp signal
for m = 1 : length(h0)
    if (m == sel_idx(cn))
        h_val = h0(m);
        if cn < length(sel_idx)
            cn = cn + 1;
        end
    end
    h(m) = h_val;
end
my_h = h(1:sig_len);


hs = square(2 * pi * f0 .* t + pi * c .* t .* t + p, 20) + 1;
count_thres = floor(0.9 * Fs / (f0 + c * t(1)));
count = count_thres + 1;
for m = 1:length(hs)
    if (hs(m) == 0)
        count_thres = floor(0.9 * Fs / (f0 + c * t(m)));
        count = count_thres + 1; % reset count threshold
    else
        if count >= count_thres
            hs(m) = h(m);
            count = 1;
        else
            count = count + 1;
            hs(m) = 0;
        end
    end
    
end
hs(find(hs > 1)) = 2;

my_hs = hs(1 : sig_len);
erp_len = round(1.0 / 8 * Fs);

% shift
for k=1:erp_len
    H(k, k : k + sig_len-1) = hs;
end

H = H(:, 1:sig_len);

end
