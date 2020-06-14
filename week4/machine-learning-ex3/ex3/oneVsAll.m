function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  %ONEVSALL trains multiple logistic regression classifiers and returns all
  %the classifiers in a matrix all_theta, where the i-th row of all_theta 
  %corresponds to the classifier for label i
  %   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
  %   logistic regression classifiers and returns each of these classifiers
  %   in a matrix all_theta, where the i-th row of all_theta corresponds 
  %   to the classifier for label i


  dataSetSize = size(X, 1);
  features = size(X, 2);
  all_theta = zeros(num_labels, features + 1);

  % Add ones to the X data matrix
  X = [ones(dataSetSize, 1) X];
  
  initial_theta = zeros(features + 1, 1);
  
  % Set options for fminuncg
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  
  for c = 1:num_labels
    [theta, cost] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
    all_theta(c,:) = theta(:);
  endfor
end
