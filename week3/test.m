clear;
clc;

function value = sigmoidFunction(matrix)
  value = (1 ./ (1 + exp(-matrix)));
endfunction

function J = cost(theta, X, y)
  trainingExamples = length(y);
  J = (1 / (2 * trainingExamples)) * sum((sigmoidFunction(X * theta) - y) .^ 2);
endfunction

function J = regularizedCost(theta, X, y)
  trainingExamples = length(y);
  regularizationParameter = 100;
  
  J = (1 / (2 * trainingExamples)) * (
      sum((sigmoidFunction(X * theta) - y) .^ 2)
      + regularizationParameter * (sum(theta .^ 2) - theta(1) ^ 2)
  );
endfunction

function gradient = gradientVector(theta, X, y)
  trainingExamples = length(y);
  gradient = (1 / trainingExamples) * (X' * (sigmoidFunction(X * theta) - y));
endfunction

function gradient = regularizedGradientVector(theta, X, y)
  trainingExamples = length(y);
  regularizationParameter = 100;
  gradient = (1 / trainingExamples) * (
    X' * (sigmoidFunction(X * theta) - y) 
    + regularizationParameter * theta
  );
endfunction

function [value, gradient] = optimizationFunction(theta)
  data = [1 1 ; 1 2 ; 1 3];
  result = [1 ; 2 ; 3];
  value = cost(theta, data, result);
  gradient = gradientVector(theta, data, result);
endfunction

function [theta, costMemory, minCost] = gradientDescent(theta, X, y, iterations, learningRate)
  costMemory = [cost(theta, X, y)];
  for i = 1:iterations
    theta = theta - learningRate * gradientVector(theta, X, y);
    costMemory = [costMemory cost(theta, X, y)];
  end
  minCost = cost(theta, X, y);
endfunction

data = [1 1 ; 1 2 ; 1 3];
result = [1 ; 2 ; 3];
hypothesis = [10 ; 0];

[theta, costMemory, minCost] = gradientDescent(hypothesis, data, result, 100, 0.03);
disp(theta);
disp(minCost);
plot(costMemory);
