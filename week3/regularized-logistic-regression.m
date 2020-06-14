clc;
clear;
close;

function lambda = regularizationFactor()
  lambda = 10;
endfunction

function value = sigmoid(vector)
  value = 1 ./ (1 + exp(-vector));
endfunction

function J = logisticRegressionCost(theta, X, y)
  estimatedResults = sigmoid(X * theta);
  trainingSamples = length(y);
  J = -(1 / trainingSamples) * sum(
    y .* log(estimatedResults) 
    + (1 - y) .* log(1 - estimatedResults)
  );
endfunction

function J = logisticRegressionRegularizedCost(theta, X, y)
  estimatedResults = sigmoid(X * theta);
  trainingSamples = length(y);
  
  J = (- 1 / trainingSamples) * sum(
    y .* log(estimatedResults) 
    + (1 - y) .* log(1 - estimatedResults)
  ) + (regularizationFactor() / (2 * trainingSamples)) * (
    sum(theta .^ 2) - theta(1) ^ 2
  );
endfunction

function gradient = gradientVector(theta, X, y)
  trainingExamples = length(y);
  gradient = (1 / trainingExamples) * (X' * (sigmoidFunction(X * theta) - y));
endfunction

function gradient = regularizedGradientVector(theta, X, y)
  trainingExamples = length(y);
  gradient = gradientVector(theta, X, y);
  modifiedHypothesis = (regularizationFactor() / trainingExamples) * theta;
  modifiedHypothesis(1) = 0;
  gradient += modifiedHypothesis;
endfunction

function [theta, costMemory, minCost] = gradientDescent(theta, X, y, iterations, learningRate)
  costMemory = [logisticRegressionCost(theta, X, y)];
  for i = 1:iterations
    theta = theta - learningRate * gradientVector(theta, X, y);
    costMemory = [costMemory logisticRegressionCost(theta, X, y)];
  endfor
  minCost = logisticRegressionCost(theta, X, y);
endfunction

function [theta, costs, minCost] = regularizedGradientDescent(theta, X, y, iterations, learningRate)
  costs = [logisticRegressionRegularizedCost(theta, X, y)];
  for i = 1:iterations
    theta = theta - learningRate * regularizedGradientVector(theta, X, y);
    costs = [costs logisticRegressionRegularizedCost(theta, X, y)];
  endfor
  minCost = logisticRegressionRegularizedCost(theta, X, y);
endfunction

hypothesis = [0 ; 0];
data = [1 1 ; 1 2 ; 1 3];
result = [1 ; 1 ; 1];

[theta, costMemory, minCost] = gradientDescent(hypothesis, data, result, 3000, 0.05);
% disp(theta);
disp(minCost);
subplot(2, 2, 1); plot(costMemory);
subplot(2, 2, 2); imagesc(theta);

[theta, costMemory, minCost] = regularizedGradientDescent(hypothesis, data, result, 3000, 0.05);
% disp(theta);
disp(minCost);
subplot(2, 2, 3); plot(costMemory);
subplot(2, 2, 4); imagesc(theta);
