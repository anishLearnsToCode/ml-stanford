function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

  function J = logisticRegressionCost(theta, X, y)
    estimatedResults = sigmoid(X * theta);
    trainingSamples = length(y);
    J = -(1 / trainingSamples) * (
      y' * log(estimatedResults) 
      + (1 - y)' * log(1 - estimatedResults)
    );
  endfunction

  function gradient = gradientVector(theta, X, y)
    trainingExamples = length(y);
    gradient = (1 / trainingExamples) * (X' * (sigmoid(X * theta) - y));
  endfunction

  J = logisticRegressionCost(theta, X, y);
  grad = gradientVector(theta, X, y);
end
