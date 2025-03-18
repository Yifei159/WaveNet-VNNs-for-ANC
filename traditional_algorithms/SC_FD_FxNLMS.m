function [e, W] = SC_FD_FxNLMS(x, d, secpath, con_len, mu, delta, eta)
%% Single-Channel-Frequency-Domain FxNLMS Algorithm 
% [x]  Reference signal (T × 1)
% [d]  Desired signal (T × 1)
% [secpath]  Secondary path (N × 1)
% [con_len]  Control filter length
% [mu]  Convergence step size
% [delta]  Normalization factor
% [eta]  Nonlinear parameter (represents η²)
%% Parameters
secpath0 = secpath; 
sec_len = size(secpath, 1);
N = size(x, 1);  
%% Convolution Buffer and Filter Coefficient Buffer
W = zeros(con_len, 1);  
y = zeros(N,1);  
s = zeros(N, 1);  
e = zeros(N, 1);  
%% Frequency domain parameters
data_size = con_len;
block_size = 2 * data_size;
ref_buffer = zeros(con_len, 1);  
filter_ref_buffer = zeros(sec_len, 1); 
output_buffer = zeros(sec_len, 1);  
filter_ref_buffer_block = zeros(block_size, 1);  
e_block = zeros(block_size, 1);  
px4 = ones(block_size, 1);  
mu_block = mu * ones(block_size, 1);  
delta_block = delta * ones(block_size, 1);  
pert = 0;
%% FxLMS
for n = 1:N
    if(length(x)*pert/100<n)
       pert = pert + 1;
       fprintf('%2d%%',pert);
    end
    ref_buffer = [x(n); ref_buffer(1:con_len-1)];
    filter_ref_buffer = [x(n); filter_ref_buffer(1:sec_len-1)];
    Wf_temp = W;  
    y(n) = Wf_temp.' * ref_buffer; 
    if eta~=0
       y(n) = sef(y(n),eta);
    end
    output_buffer = [y(n); output_buffer(1:sec_len-1)];
    S_temp = secpath0;
    s(n) = S_temp.' * output_buffer;
    %superpositon of the primary noise and control signal
    e(n) = d(n) + s(n);
    %=================feedforward update====================
    e_block(1:end-1) = e_block(2:end);
    e_block(end) = e(n);
    feedforward_filtered_ref = secpath.' * filter_ref_buffer;  
    filter_ref_buffer_block(1:end-1) = filter_ref_buffer_block(2:end);
    filter_ref_buffer_block(end) = feedforward_filtered_ref;
    if mod(n, data_size) == 0
        ef = fft([zeros(data_size, 1); e_block(data_size+1:end)]);  
        rf = fft(filter_ref_buffer_block);
        wf_buffer = conj(rf) .* ef;  
        wf = wf_buffer;  
        px4(:) = conj(rf) .* rf;
        %feedforward con_filter update
        wt1 = mu_block .* wf ./ (px4 + delta_block);
        wt = ifft(wt1);  
        W(1:data_size) = W(1:data_size) - real(wt(1:data_size));
    end
end
end