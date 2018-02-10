%%
%Prepare Data
% t = rand(10000, 1);
% input = exp(-abs(t).^1.5); %Input signal
noise = wgn(10000, 1, 5.0); %noise
% 
% b = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1];
% a = [1, -1];
% output = filter(b, a, input);
y_n = output + noise;  %Output with noise

% Fs = 1000;
% N = length(input);
% xdft = fft(input);
% xdft = xdft(1:N/2+1);
% psdx = (1/(Fs*N)) * abs(xdft).^2;
% psdx(2:end-1) = 2*psdx(2:end-1);
% freq = 0:Fs/length(input):Fs/2;
% 
% plot(freq,10*log10(psdx))
% grid on
% title('Input PSD')
% xlabel('Frequency (Hz)')
% ylabel('Power/Frequency (dB/Hz)')

M = 15; %Filter Order
window_size = 100;
w = rand(M,1);
%eta = 0.00001; %Learning rate

x = zeros(10000, M);  %With zero pad
for m = M:-1:1
    for i = 1:(10000-M+m)
        x(i+M-m, m) = input(i);
    end
end
MSE = zeros(10000/window_size,1); 
pred_save = zeros(10000,1);
WSNR = zeros(10000/window_size,1);

w_star = [1 1 1 1 1 1 1 1 1 1]';  %INV(R)*P

% for batch = 1:(10000/window_size)
%     x_batch = x((1+(batch-1)*window_size):(window_size*batch), :);
%     R = x_batch' * x_batch;  %Auto-correlation matrix
%     P = x_batch' * y_n((1+(batch-1)*window_size):(batch*window_size));
%     w = R\P;  %INV(R)*P
%     wp = zeros(30,1);
%     wp(1:10) = w_star;
%     WSNR(batch) = 10*log((wp'*wp)/((wp-w)'*(wp-w)));
% end

w_save = zeros(10000/window_size, M);
for batch = 1:(10000/window_size)
    x_batch = x((1+(batch-1)*window_size):(window_size*batch), :);
    R = x_batch' * x_batch;  %Auto-correlation matrix
    P = x_batch' * y_n((1+(batch-1)*window_size):(batch*window_size));
    w = R\P;  %INV(R)*P
    w_save(batch,:) = w;
    pred = x_batch * w;  %Predict value of output
    pred_save(window_size*(batch-1)+1:window_size*batch) = pred;
    err = y_n((1+(batch-1)*window_size):(batch*window_size)) - pred;
    MSE(batch) = (err' * err)/(trace(x_batch'*x_batch)*window_size);
    %batch
end

plot(w_save); hold on;
grid on
title('Weight track over different noise')
xlabel('Window')
ylabel('Weight')
% 
% plot(WSNR); hold on;
% plot(y_n); hold;
% plot(pred_save);
%sound(data, fs);
% max(WSNR)