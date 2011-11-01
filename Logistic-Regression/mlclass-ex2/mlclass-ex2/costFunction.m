function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% NOTES:
% From dataset 'ex2data1.txt' we expect that:
% X 	= 100 x 3 matrix ... these are the training examples (inputs)
% theta = 3 x 1 matrix 	 ... these are the 'weights' applied to EACH example in X
% y  	= 100 x 1 matrix ... this is the decision matrix [0-1] (outputs)

% Hypothesis (prediction): 100 x 1 matrix. 
% matrix size of h should match y
h = sigmoid(X*theta);

% ----------------------
% NON-VECTORIZED VERSION
% Cost matrix: 100 x 1 matrix
% J(theta): 1 x 1
% ----------------------
% Cost = zeros(size(h));
% for i = 1:length(y)
% 	Cost(i) = (y(i) * log(h(i))) + ((1-y(i)) * log(1-h(i)));
% end
%
% J = -1/m * sum(Cost);
% ----------------------

% ----------------------
% VECTORIZED VERSION
% Cost matrix: 100 x 1 matrix
% J(theta): 1 x 1
% ----------------------
Cost = (y' * log(h)) + ((1-y')*log(1-h));
J = -1/m * Cost;

% Gradient: partial derivative of J with respect to theta

% for i = 1:length(theta)
% 	grad(i) = 1/m * sum((h - y) .* X(:,i));
% end
grad = 1/m * (X' * (h - y));

% =============================================================

end
