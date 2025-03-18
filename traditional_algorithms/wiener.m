function [W, e] = wiener(ref, E, filter_len, sec_path, beta, eta, ref_test, E_test)
% Single-channel Wiener filtering algorithm.
%==============INPUT================
% ref: Reference signal.
% E  : Primary noise signal.
% filter_len: Control filter length.
% sec_path: Secondary path transfer function.
% beta: Regularization factor, chosen based on noise reduction and stability.
% ref_test: To avoid overfitting, a separate reference signal used for noise reduction evaluation 
%           (can be ignored if overfitting is not considered).
% E_test: To avoid overfitting, a separate primary noise signal used for noise reduction evaluation.

%==============OUTPUT================
% W: Control filter coefficients.
% e: Residual error signal.

x = filter(sec_path, 1, ref);
r = xcorr(x, x, filter_len - 1); 
r = flipud(r(1:filter_len));

reg = zeros(filter_len, 1); % Regularization
reg(1) = beta * eye(1);
R = r + reg;

p = xcorr(E, x, filter_len - 1);
P = p(filter_len : 2 * filter_len - 1)';

W = block_levinson(P, R);
if nargin < 7
    ref_test = ref;
    E_test = E;
end
xx = filter(W, 1, ref_test);
if eta ~= 0
    xx = sef(xx, eta);
end
yy = filter(sec_path, 1, xx);
e = E_test - yy;
