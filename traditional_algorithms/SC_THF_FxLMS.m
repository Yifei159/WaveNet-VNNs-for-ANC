function [e,W] = SC_THF_FxLMS(x,d,secpath,con_len,normalized,mu,delta,eta)
%% Single-Channel-Tangential hyperbolic function FxLMS Algorithm 
% normalized == 0: FxLMS / normalized == 1: NFxLMS
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
N = size(x,1); 
alpha = sqrt(2*eta);
beta = 1/alpha;
%% Convolution Buffer and Filter Coefficient Buffer
W = zeros(con_len,1);  
y = zeros(N,1);  
s = zeros(N,1);     
e = zeros(N,1);     
ref_buffer = zeros(con_len,1);  
filter_ref_buffer = zeros(sec_len,1);  
output_buffer = zeros(sec_len,1);  
filtered_ref_buffer = zeros(con_len,1); 
pert = 0;
%% FxLMS
for n = 1:N
   if n < 513
       delta = 0.1;
   end
   if(length(x)*pert/100<n)
       pert = pert + 1;
       fprintf('%2d%%',pert);
    end
    %processing the reference signal for the feedforward part
    ref_buffer = [x(n); ref_buffer(1:con_len-1)];
    filter_ref_buffer = [x(n); filter_ref_buffer(1:sec_len-1)];
    Wf_temp = W; 
    y(n) = Wf_temp.'*ref_buffer;%feedforward output calculation
    if eta~=0
       y(:,n) = sef(y(:,n),eta);
    end
    THF_Y = alpha*beta*(1-tanh(beta*y(n))^2);
    %propagate through the secondary path 
    output_buffer = [y(n); output_buffer(1:sec_len-1)];
    S_temp = secpath0;
    s(n) = S_temp.'*output_buffer;
    %superpositon of the primary noise and control signal
    e(n) = d(n) + s(n);
    %=================feedforward update====================
    feedforward_filtered_ref = secpath'*filter_ref_buffer;
    filtered_ref_buffer = [feedforward_filtered_ref; filtered_ref_buffer(1:con_len-1)];
    %feedforward con_filter update
    if normalized == 0         
        W = W - mu*e(n)*THF_Y*filtered_ref_buffer; 
    end
    if normalized == 1
        nfactor_f = update_tmp_f.'*update_tmp_f;  
        normalized_miu_f = mu/(nfactor_f + delta); 
        W = W - normalized_miu_f*e(n)*THF_Y*filtered_ref_buffer;
    end
 end 
end