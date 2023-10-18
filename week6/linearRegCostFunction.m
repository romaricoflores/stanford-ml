function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate the hypothesis
h = X * theta;

% Calculate the cost function (without regularization)
J = (1 / (2 * m)) * sum((h - y) .^ 2);

% Calculate the regularization term
reg_term = (lambda / (2 * m)) * sum(theta(2:end) .^ 2);

% Add regularization term to the cost function
J = J + reg_term;

% Calculate the gradient (without regularization)
grad = (1 / m) * X' * (h - y);

% Calculate the regularization term for gradient
reg_grad = (lambda / m) * [0; theta(2:end)];

% Add regularization term to the gradient
grad = grad + reg_grad;

% =========================================================================

grad = grad(:);

end
