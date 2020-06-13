function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

  trainingExamles = size(X)(1);
  predictions = X * theta;
  squareErrors = (predictions - y) .^ 2;
  J = 0.5 / trainingExamles * sum(squareErrors);
end
