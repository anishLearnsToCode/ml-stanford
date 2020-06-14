function [J, grad] = lrCostFunction(theta, X, y, lambda)
  %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  %regularization
  %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters. 
  
  function J = logisticRegressionRegularizedCost(theta, X, y)
    estimatedResults = sigmoid(X * theta);
    trainingExamples = length(y);
    
    J = (- 1 / trainingExamples) * (
      y'  * log(estimatedResults) 
      + (1 - y)' * log(1 - estimatedResults)
    ) + (lambda / (2 * trainingExamples)) * (
      sum(theta .^ 2) - theta(1) ^ 2
    );
  endfunction
  
  function gradient = gradientVector(theta, X, y)
    trainingExamples = length(y);
    gradient = (1 / trainingExamples) * (X' * (sigmoid(X * theta) - y));
  endfunction
  
  function gradient = regularizedGradientVector(theta, X, y)
    trainingExamples = length(y);
    gradient = gradientVector(theta, X, y);
    modifiedHypothesis = (lambda / trainingExamples) * theta;
    modifiedHypothesis(1) = 0;
    gradient += modifiedHypothesis;
  endfunction

  J = logisticRegressionRegularizedCost(theta, X, y);
  grad = regularizedGradientVector(theta, X, y);
end
