clc;
clear;
close all;

% Combined Collaborative Filtering (Low rank marix factorization algo)

function J = combinedCollaborativeFiltering(X, y, thetas, r)
  J = combinedLinearRegressionCost(X, y, thetas, r) + regularizedCost(X) 
    + regularizedCost(thetas);
endfunction

% @return J:scalar cost
% @param X:(n_m),(n) feature vector per movie
% @param y:(n_m),(n_u) rating per user for a given movie 
% @param thetas:(n_u),(n) theta vector per user
% @param r:(n_m),(n_u) boolean matrix whether user has rated a movie or not (1/0)
% @param userId:scalar user id
% n_m: number of movies n_u: number of users
function J = linearRegressionCost(X, y, thetas, r, userId)
  moviesSeen = find(r(:, userId) == 1);
  X = X(moviesSeen, :);
  [m, n] = size(X); 
  y = y(moviesSeen, userId);
  theta = thetas(userId, :)';
  J = 0.5 * sum((X * theta - y) .^ 2);
endfunction

function J = combinedLinearRegressionCost(X, y, thetas, r)
  J = 0;
  [numberOfMovies, numberOfUsers] = size(y);
  for i = 1:numberOfUsers
    J += linearRegressionCost(X, y, thetas, r, i);
  endfor
endfunction

function J = regularizedCost(X)
  J = (regularizationFactor() / 2) * (sum(sum(X .^ 2)) - sum(X(1) .^ 2)); 
endfunction

function lambda = regularizationFactor()
  lambda = 5;
endfunction

function g = gradientTheta(X, y, thetas)
  g = (X * thetas' - y)' * X + regularizationFactor() * thetas;
endfunction

function g = gradientX(X, y, thetas, r)
  g = (X * thetas' - y) * thetas + regularizationFactor() * X;
endfunction

function [minCost, thetas, X, costs] = gradientDescent(X, y, thetas, r, iters, alpha)
  costs = [];
  for i = 1:iters
    X = X - alpha * gradientX(X, y, thetas);
    thetas = thetas - alpha * gradientTheta(X, y, thetas);
    minCost = combinedCollaborativeFiltering(X, y, thetas, r);
    costs = [costs minCost];
  endfor
endfunction

function [minCost, thetas, X, costs] = initializeCollaborativeFiltering(
  numberOfMovies, numberOfUsers, features)
  
  X = rand(numberOfMovies, features);
  y = round(5 * rand(numberOfMovies, numberOfUsers));
  thetas = rand(numberOfUsers, features);
  r = rand(numberOfMovies, numberOfUsers) >= 0.5;
  [minCost, thetas, X, costs] = gradientDescent(X, y, thetas, r, 20, 0.03);
endfunction

[minCost, thetas, X, costs] = initializeCollaborativeFiltering(5, 3, 1);
