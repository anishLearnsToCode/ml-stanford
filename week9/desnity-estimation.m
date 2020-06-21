clc;
clear;
close all;

function g = gaussian(x, mu, variance)
  g = ((x - mu) .^ 2) ./ (2 * variance);
  g = exp(-g);
  % factor = 1 ./ sqrt(2 * pi * variance);
  % g .*= factor;
endfunction

function g = gaussianMat(x)
  mu = mean(x);
  variance = var(x);
  g = gaussian(x, mu, variance);
endfunction

function p = gaussianProbabilities(x)
  g = gaussianMat(x);
  p = prod(g, 2);
endfunction

function p = probabilityGaussianModel(x, mu, variance)
  g = gaussian(x, mu, variance);
  p = prod(g, 2);
endfunction

function y = isAnomaly(x, mu, variance, threshold)
  p = probabilityGaussianModel(x, mu, variance);
  y = p < threshold;
endfunction

function [mu, variance] = gaussianModel(x)
  mu = mean(x);
  variance = var(x);
endfunction

function [precision, recall, f1] = evaluationMetrics(computed, actual)
  truePositives = sum(computed + actual == 2);
  totalPositives = sum(computed);
  actualPositives = sum(actual);
  precision = truePositives / totalPositives;
  recall = truePositives / actualPositives;
  f1 = (2 * precision * recall) / (precision + recall);
endfunction

function [mu, covMatrix] = multivariateGaussianModel(x)
  mu = mean(x);
  m = size(x, 1);
  covMatrix = (1 / m) * x' * x;
endfunction

function p = probabilityMultivariateGaussian(x, mu, covMatrix)
  n = size(x, 2);
  g = -(1 / 2) * (x - mu) * pinv(covMatrix) * (x - mu)';
  g = exp(-g);
  detCovMatrix = det(covMatrix);
  factor = (detCovMatrix ^(1 / 2)) * ((2 * pi) ^ (n / 2));
  p = diag(g ./ factor);
endfunction

x = [1 2 3 4 5 ; 3 4 5 6 7 ; 0 0 0 3 10];
xCrossValidation = [0 0 -90 67 3 ; 3 5 6 0 23];
[mu variance] = gaussianModel(x);
disp(probabilityGaussianModel(x, mu, variance));
[mu, covMatrix] = multivariateGaussianModel(x);
disp(probabilityMultivariateGaussian(x, mu, covMatrix));
 