clc;
clear;

function g = cost1(z)
  mask = z < 1;
  g = z - 1;
  g .*= mask;
  g = abs(g);
endfunction

function g = cost0(z)
  mask = z > -1;
  g = z + 1;
  g .*= mask;
  g = abs(g);
endfunction

function J = svmCost(theta, X, y, C)
  probabilities = X * theta;
  J = y .* cost1(probabilities) + (1 - y) .* cost0(probabilities);
  J *= C;
endfunction

function J = regularizedWeights(theta)
  J = sum(theta .^ 2) - theta(1) ^ 2;
endfunction

function J = svmRegularizedCost(theta, X, y, C) 
  J = svmCost(theta, X, y, C) + (1 / 2) * regularizedWeights(theta);
endfunction

% Gaussian Kernel Function
function f1 = similarity(X, l1)
  trainingDataSize = size(X, 1); 
  similarityVector = repelem(l1, trainingDataSize, 1);
  difference = X - similarityVector;
  euclideanDistanceSquared = sum(difference .^ 2, 2);
  variance = 1;
  f1 = exp(- euclideanDistanceSquared ./ (2 * variance));
endfunction

function k = kernel(X, l)
  trainingSampleSize = size(X, 1);
  anchorPoints = size(l, 1);
  k = zeros(trainingSampleSize, anchorPoints);
  for i = 1:anchorPoints
    anchorVector = l(i, :);
    k(:, i) = similarity(X, anchorVector);
  endfor
endfunction

function p = predictKernel(theta, X)
  probabilities = X * theta;
  p = probabilities >= 0;
endfunction

mat = [1 2 3 0 -1 -2 -3];
% disp(cost0(mat));
% disp(cost1(mat));

X = [1  1 ; 1 2 ; 1 3];
l1 = [10 10];
l = [10 10 ; 5 6 ; -10 -8 ; 0 0 ; 100 100];
% disp(similarity(X, l1));
disp(kernel(X, X));
disp(predictKernel([1 ; 2], [1 1 ; 1 2 ; 1 3 ; 1 -10]));


% disp(var([1, 2, 3, 4, 10]));
% disp(var(X, [], 2));

mat = [ 1 2 3 ; 10 -89 0];
[min, index] = min(mat(:, 3));
disp(min);
disp(index);
