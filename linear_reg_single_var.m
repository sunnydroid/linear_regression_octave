% workbook to practice and visualize gradient descent

% loading and plotting data
data = load('ex1data1.txt');
% plot population vs profits graph
x = data(:, 1);
y = data(:, 2);
%sample size 
m = length(y);

plot(x, y, 'rx', 'markersize', 10) 
title('Population vs Profit')
xlabel('Population in 10,000')
ylabel('Profit in $10,000')

% model linear regression y = b + Tx
X = [ones(m, 1), data(:, 1)]; % add a column of 1s to x as the intercept term for each sample
% initialize theta vector to (0, 0)'
theta = zeros(2, 1);

% cost or object function is the function that calculates the difference between 
% the true/expected output and calculated output
% cost = 1/2m * sum_over_all_samples(calculated_output - actual_output)^2
function average_cost = computeCost(X, y, theta)
	% the square operation for the vectrized form can be easily calculated using transpose.
	% transpose of a matrix is performed by the ' operator
	m = length(y);
	cost = (X*theta - y);
	% initial cost with theta = zeros
	average_cost = (cost' * cost) / (2*m);
end

initial_cost = computeCost(X, y, theta);
% Gradient descent simultaneously updates theta parameters with the objective of finding theta values
% that reduce the cost function. During each iteration, each theta value is updated simultaneously, being
% reduced by derative of the function at that point - which translate to reduction of theta by the contribution 
% of sample x(i) to the cost at that particular point 
% theta0 = theta0 - partial derivative of cost function w.r.t theta0
% similarly
% theta1 = theta1 - partial derivative of cost function w.r.t theta1
% which reduces to 
% theta1 = theta1 - (cost * x(i))
iterations = 1500;
learning_rate = 0.01;

% keep track of cost at each iteration so we can visualize it in cost vs iteration plot
J = zeros(iterations, 1);

for iter = 1:iterations
	cost_iter = (X*theta - y);
	rms_cost = (cost_iter' * cost_iter) * (1/(2*m));
	% fprintf('cost during iteration %f cost is %f\n', iter, rms_cost);
	cost_derivative_theta1 = sum(cost_iter) / m;
	% cost derivative for theta2 has to be calculated element wise, i.e. elementwise square and multiplication 
	% by contribution of that feature at that sample
	cost_derivative_theta2 = ((cost_iter)' * x) / m;
	% adjust each theta value by the learning rate x cost derivatives w.r.t each theta
	% note: indices in matlab/octave begin at 1 and not 0
	theta(1) = theta(1) - (learning_rate * cost_derivative_theta1);
	theta(2) = theta(2) - (learning_rate * cost_derivative_theta2);

	% fprintf('after iteration %d new theta values are %f and %f\n', iter, theta(1), theta(2));
	J(iter) = rms_cost;

	% all the steps above can be condensed to a single line due to vector/matrix form representation
	% theta = theta - X' * (X*theta - y) * learning_rate * (1/m);
end

hold on;
plot(X(:, 2), X*theta, 'b-');
legend('training data', 'linear regression');
hold off;

figure;
plot(1:1:iter, J, 'b-');
title('Cost Vs Iteration');
legend('cost');
xlabel('iteration');
ylabel('cost');

% In order to plot the contor and surface plots of the cost function
% we need to calculate the cost over a range of values for theta1 vs theta2
% define linear space for range of values
theta1_range = linspace(-10,10,100);
theta2_range = linspace(-1,4,100);
%initialize cost matrix that will store costs over the range of theta1 and theta2
costs = zeros(length(theta1_range), length(theta2_range));

for i=1:length(theta1_range)
	for j=1:length(theta2_range)
		% build theta vector to compute cost with 
		thetaVector = [theta1_range(i); theta2_range(j)];
		costs(i, j) = computeCost(X, y, thetaVector);
	end
end

% surface plot
figure;
% cost needs to be transposed before it can be plotted on mesh grid
costs = costs';
surf(theta1_range, theta2_range, costs);
xlabel('theta0');
ylabel('theta1');
title('Cost Function Surface Plot');

% contour plot
figure;
contour(theta1_range, theta2_range, costs, logspace(-2, 3, 20));
xlabel('theta0');
ylabel('theta1');
title('Contour Plot - theta0 vs theta 1');
% overlay computed values of theta1 and theta2
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2) 
