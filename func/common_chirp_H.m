function [common_H] = common_chirp_H(H, f0, c, Fs)
% H : original chirp impulse
% f0: initial frequency
% f : final frequency
% c : chirp rate

[I, J] = size(H);
common_H = zeros(I, J);
for i = 1 : I % row
    for j = i : J % colume
        trans = 8.0 / (f0 + c * (j - i) / Fs); % f0 + c*t

        % if c < 0
        %     trans = 1 / trans;
        % end

        if i == 1 || common_H(i - 1, j) == 0
            common_H(i, j) = H(ceil(trans * i), j);
        end
        % if trans * i <= size(H, 1)
        %     common_H(i, j) = H(round(trans * i), j);
        % else
        %     common_H(i, j) = 0; % Avoid index out of bounds
        % end
    end
end

end