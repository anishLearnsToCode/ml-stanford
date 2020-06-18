function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  %regression with multiple variables
  %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  %   cost of using theta as the parameter for linear regression to fit the 
  %   data points in X and y. Returns the cost in J and the gradient in grad

  m = length(y); % number of training examples
  features = size(X, 2) - 1;

  function cost = regressionCost(theta, X, y)
    cost = (1 / (2 * m)) * sum((X* theta - y) .^ 2);
  endfunction

  function cost = regularizedRegressionCost(theta, X, y)
    cost = regressionCost(theta, X, y);
    cost += (lambda / (2 * m)) * (sum(theta .^ 2) - theta(1)^2);
  endfunction
  
  J = regularizedRegressionCost(theta, X, y);
  
  function grad = regressionGradient(theta, X, y)
    grad = (1 / m) * ((X * theta - y)' * X)';
  endfunction

  function grad = regularizedRegressionGradient(theta, X, y)
    grad = regressionGradient(theta, X, y);
    regularizationMask = (lambda / m) * ones(features + 1, 1);
    regularizationMask(1) = 0;
    grad += regularizationMask .* theta;
  endfunction
  
  grad = regularizedRegressionGradient(theta, X, y);

  % converting into column vector
  grad = grad(:);
end
