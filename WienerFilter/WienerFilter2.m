%%
d = zeros(18000,1);
for i = 1:18000
    d(i) = data(i);
end
M = 15; %Filter Order
[N,n] = size(d);
window_size = 100;
w = rand(M,1);

%Predict next value of input
D = zeros(18000,1);
for i = 1:18000
    D(i) = data(i+1);
end

x = zeros(N, M);  %With zero pad
for m = M:-1:1
    for i = 1:(N-M+m)
        x(i+M-m, m) = d(i);
    end
end
MSE = zeros(N/window_size,1); 
pred_save = zeros(N,1);

w_save = zeros(N/window_size, M);
for batch = 1:(N/window_size)
    x_batch = x((1+(batch-1)*window_size):(window_size*batch), :);
    pred = x_batch * w;  %Predict value of output
    R = x_batch' * x_batch;  %Auto-correlation matrix
    P = x_batch' * D((1+(batch-1)*window_size):(batch*window_size));
    w = R\P;  %INV(R)*P
    w_save(batch,:) = w;
    pred = x_batch * w;  %Predict value of output
    pred_save(window_size*(batch-1)+1:window_size*batch) = pred;
    err = D((1+(batch-1)*window_size):(batch*window_size)) - pred;
    MSE(batch) = (err' * err)/(trace(x_batch'*x_batch));
    %batch
end

% plot(MSE); hold on;
% grid on
% title('Normalized MSE')
% xlabel('Window')
% ylabel('Prediction Error')

plot(w_save); hold on;
grid on
title('Weight Track')
xlabel('Window')
ylabel('Weight')

% sound(pred_save,fs);