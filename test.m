clc;
clear;

function error = logisticMisclassificationError(hypothesis, y, threshold)
  prediction = hypothesis >= threshold;
  error = sum(prediction != y) / length(y);
endfunction

function J = regressionCost(hypothesis, data, result)
  trainingSampleSize = size(data, 1);
  J = (1 / (2 * trainingSampleSize)) * sum((data * hypothesis - result) .^ 2);
endfunction

function J = regularizedRegressionCost(theta, X, y, regularizationConstant)
  J = regressionCost(theta, X, y);
  trainingSampleSize = length(y);
  J += (regularizationConstant / (2 * trainingSampleSize)) * sum(theta .^ 2);
endfunction

function grad = regressionGradient(hypothesis, X, y)
  trainingSampleSize = size(X, 1);
  grad = (1 / trainingSampleSize) * (((X * hypothesis) - y)' * X)';
endfunction

function grad = regularizedRegressionGradient(theta, X, y, lambda)
  grad = regressionGradient(theta, X, y);
  trainingSampleSize = size(X, 1);
  features = size(X, 2) - 1;
  regularizationMask = (lambda / trainingSampleSize) * ones(features + 1, 1);
  regularizationMask(1) = 0;
  grad += regularizationMask .* theta;
endfunction

function [minCost, hypothesis, costMemory] = gradientDescent(hypothesis, X, y, iterations, learningRate)
  costMemory = [];
  for i = 1:iterations
    hypothesis = hypothesis - learningRate * regressionGradient(hypothesis, X, y);
    minCost = regressionCost(hypothesis, X, y);
    costMemory = [costMemory minCost];
  endfor
endfunction

hypothesis = [0.3 ; 0.5 ; 0.3 ; 0.9];
result = [1 ; 0 ; 1 ; 1];
error = logisticMisclassificationError(hypothesis, result, 0.5);
disp(error);

hypothesis = [0 ; 3];
data = [1 1 ; 1 2 ; 1 3];
y = [1 ; 2 ; 3];
disp(regressionCost(hypothesis, data, y, 3));

initialTheta = [0 ; 0 ];
[minCost, hypothesis, costMemory] = gradientDescent(initialTheta, data, y, 100, 0.09);
disp('min cost');
disp(minCost);

disp('hypothesis');
disp(hypothesis);

plot(costMemory);

disp('gradient');
disp(regularizedRegressionGradient(hypothesis, data, y, 10));

clc;

X = [1 1 ; 1 2 ; 1 3];
y = [1 ; 2 ; 3];
lambda = 0.05;
theta = trainLineaReg(X, y, lambda);
disp(theta);
