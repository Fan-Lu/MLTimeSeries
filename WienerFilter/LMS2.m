M = 15; %Filter Order
[N,n] = size(data);
w = 0.01*rand(M,1);
eta = 0.1; %Learning rate

x = zeros(N, M);  %With zero pad
for m = M:-1:1
    for i = 1:(N-M+m)
        x(i+M-m, m) = data(i);
    end
end
MSE = zeros(N,1); 
pred_save = zeros(N,1);

d = zeros(N,1);
for i =1:(N-1)
    d(i) = data(i+1);
end
w_save = zeros(N, M);
for batch = 1:N
    x_batch = x(batch, :);
    pred = x_batch * w;  %Predict value of output
    err = d(batch) - pred;
    pred_save(batch) = pred;
    w = w + eta*x_batch'*err;
    w_save(batch,:) = w;
    MSE(batch) = (err*err)/(x_batch*x_batch');
end

plot(MSE); hold on;
grid on
title('Normalized error power')
xlabel('Epoch')
ylabel('Prediction Error')
sound(pred_save,fs)