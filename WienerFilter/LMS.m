%%
%Prepare Data
%t = rand(10000, 1);
%input = exp(-abs(t).^1.5); %Input signal
noise = wgn(10000, 1, .1); %noise
% 
b = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1];
a = [1, -1];
output = filter(b, a, input);
y_n = output + noise;  %Output with noise

fid = fopen('output.txt', 'w');
fprintf(fid, '%f\r\n', output);
fclose(fid);


M = 5; %Filter Order
window_size = 1;
w = 0.01*rand(M,1);
eta = 0.00011; %Learning rate 0.00011

x = zeros(10000, M);  %With zero pad
for m = M:-1:1
    for i = 1:(10000-M+m)
        x(i+M-m, m) = input(i);
    end
end

J = zeros(10000/window_size,1);  %Loss Function
w_save = zeros(10000/window_size, M);
error_save = zeros(10000/window_size, 1);
for batch = 1:(10000/window_size)
    x_batch = x(batch, :);
    pred = x_batch * w;  %Predict value of output
    err = y_n((1+(batch-1)*window_size):(batch*window_size)) - pred;
    J(batch) = (err'*err);%/(trace(x_batch'*x_batch)*window_size);
    w = w + eta*x_batch'*err;
    w_save(batch,:) = w;
end

mean(J)
% w_star = [1 1 1 1 1 1 1 1 1 1]';  %INV(R)*P
% WSNR = 10*log((w_star'*w_star)/((w_star-w)'*(w_star-w)));

% plot(w_save); hold on;
% grid on
% title('Weight Track')
% xlabel('Epoch')
% ylabel('Weight')

% plot(J); hold on;
% title('Learning Curve')
% xlabel('Epoch')
% ylabel('MSE')
% plot(pred);