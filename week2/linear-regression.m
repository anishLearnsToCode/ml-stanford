clc;
clear;

function theta = normalizedLinearRegression(theta, X, y)
  theta = inv(X' * X) * X' * y;
endfunction

function theta = normalizedLinearRegressionWithRegularization(theta, X, y)
  regularizationParameter = 100;
  features = size(X)(2) - 1;
  regularizationMatrix = regularizationParameter * eye(features + 1);
  regularizationMatrix(1, 1) = 0;
  theta = inv(X' * X + regularizationMatrix) * X' * y;
endfunction

hypothesis = [0 ; 0];
data = [1 1 ; 1 2 ; 1 3];
result = [1 ; 2 ; 3];
optimizedHypothesis = normalizedLinearRegression(hypothesis, data, result);
disp(round(optimizedHypothesis));

optimizedHypothesis = normalizedLinearRegressionWithRegularization(hypothesis, data, result);
disp(optimizedHypothesis);
