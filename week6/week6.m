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

function precision = precision(predicted, actual)
  truePositives = sum((predicted + actual) == 2);
  predictedPositives = sum(predicted);
  precision = truePositives / predictedPositives ;
endfunction

function recall = recall(predicted, actual)
  truePositives = sum((predicted + actual) == 2);
  actualPositives = sum(actual);
  recall = truePositives / actualPositives;
endfunction

function g = sigmoid(z)
  g = 1 ./ (1 + exp(-z));
endfunction

function p = predict(theta, X)
  probabilities = sigmoid(X * theta);
  p = probabilities >= 0.5;
endfunction

% 2PR / (P + R)
function score = f1Score(precisions, recalls)
  product = precisions .* recalls;
  sum = precisions + recalls;
  score = 2 * product ./ sum;
endfunction

actual = [1 1 1 0 0 0]';
predicted = [1 0 1 1 1 0]';
disp(precision(predicted, actual));
disp(recall(predicted, actual));
disp(predict([1 ; 2], [1 1 ; 1 2 ; 1 3 ; 1 -10]));

precisions = [0.5 0.7 0.02]';
recalls = [0.4 0.1 1.0]';
disp(f1Score(precisions, recalls));
