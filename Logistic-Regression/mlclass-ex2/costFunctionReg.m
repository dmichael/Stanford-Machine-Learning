function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X*theta);
% Remember: theta(1) is not regularized
reg_theta = theta(2:end, :);

regularization = lambda/(2*m) * (reg_theta' * reg_theta);


Cost = (y' * log(h)) + ((1-y')*log(1-h));
J 	 = - (1/m * Cost) + regularization;

% Partial derivatives
grad = 1/m * (X' * (h - y));
% Regularization (except for theta(1))
grad_reg = (lambda/m)*theta;
grad_reg(1) = 0;

grad = grad + grad_reg;

% =============================================================

end
