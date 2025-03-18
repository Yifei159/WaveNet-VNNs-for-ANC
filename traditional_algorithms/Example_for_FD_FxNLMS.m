%% FD-FxNLMS-2048
clear;clc;close all;
load("pri_channel.mat");load("sec_channel.mat");
[ref, Fs] = audioread('testdata\factory1-16000nom.wav');  
len = 3744000;
ref = repmat(ref(1:len,1), 3, 1);
x = ref;
d = filter(pri_channel.',1,ref);
secpath = sec_channel.';
con_len = 2048;
mu = 0.007;
delta = 0;
eta = 0;  %Here, 0 represents ∞, which is linear.
[e,W] = SC_FD_FxNLMS(x,d,secpath,con_len,mu,delta,eta);
10.*log10(sum(e(len*2+1:end).^2)/sum(d(1:len).^2))%NMSE