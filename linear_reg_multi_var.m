data = load('ex1data2.txt');

features = (data(:, 1:rank(data)-1));
% normalize features (x - mean) / std
y = data(:, 3);
m = length(y);

% calculate mean and std for all feature columns
mu = mean(features);
sigma = std(features);

% Add intercep term to features and corresponding mu and sigma matrices 
X = [ones(m, 1), features];
% intercept term should have 0 mean and standard deviation of 1
mu = [0, mu];
sigma = [1, sigma];
% normalize feature matrix
% division by standard deviation has to be element wise
X_norm = (X - mu)./sigma;
% generalize theta to any number of features by using rank of the feature matrix 
theta = zeros(rank(X_norm), 1);

% loss function
function cost = calculateLoss(X, y, theta)
	m = length(y);
	cost = ((X*theta - y)' * (X*theta - y))/(2*m);
end

%%%%%%%%%%%%%%%%%%%% 
% gradient descent %
%%%%%%%%%%%%%%%%%%%%
learning_rate =0.01;
iterations = 1500;

% keep track of loss per iteration
J = zeros(iterations, 1);

for iter = 1:iterations
	cost = calculateLoss(X_norm, y, theta);
	J(iter) = cost;
	%fprintf('loss on iteration %.2f is %.2f\n', iter, loss);
	% vectorized implementaion X = 47 x 3, theta = 3 x 1 and y = 47 x 1
	% X * thetat - y  = (47 x 3) * (3 x 1) - (47 x 1) = 47 x 1
	% X' * (X * theta - y) = (3 x 47) * (47 x 1)  = 3 x 1
	% all values of theta are updated simultaneously
	theta = theta - (learning_rate * (1/m) * (X_norm' * (X_norm * theta - y)));
end

% closed form solution for theta can be achieved using matrix math
% y = X*theta
% multiplying both sides by X'
% X'*y = X'*X*theta
% dividing both sides by X'*X to get theta
% theta = inverse(X'*X) * (X'*y)
theta_closed_form = inverse(X' * X) * (X' * y) 
fprintf('theta learned is\n');
disp(theta);
fprintf('new values need to be normalized using mu and sigma when using learned theta needs for prediction')

plot(1:1:iter, J, 'b-');
title('Loss over each iteration');
xlabel('Iteration');
ylabel('Loss');
