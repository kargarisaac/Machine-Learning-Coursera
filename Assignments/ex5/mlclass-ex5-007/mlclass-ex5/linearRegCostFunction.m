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

H=X*theta;
theta2=theta(2:end,:);
J=(1/(2*m))*(H-y)'*(H-y) + lambda*(theta2'*theta2)/(2*m);

delta=X'*(H-y)/m;
grad(1,:)=delta(1,:);
grad(2:end,:)=delta(2:end,:) + lambda.*theta2/(m);


% =========================================================================

grad = grad(:);

end
