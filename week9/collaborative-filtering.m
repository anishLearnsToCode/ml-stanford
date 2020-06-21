clc;
clear;
close all;

% Collaborative filtering algorithm

% @return J:scalar cost
% @param X:(n_m),(n) feature vector per movie
% @param y:(n_m),(n_u) rating per user for a given movie 
% @param thetas:(n_u),(n + 1) theta vector per user
% @param r:(n_m),(n_u) boolean matrix whether user has rated a movie or not (1/0)
% @param userId:scalar user id
% n_m: number of movies n_u: number of users
function J = linearRegressionCost(X, y, thetas, r, userId)
  moviesSeen = find(r(:, userId) == 1);
  X = X(moviesSeen, :);
  [m, n] = size(X); 
  X = [ones(m, 1) X];
  y = y(moviesSeen, userId);
  theta = thetas(userId, :)';
  J = 0.5 * sum((X * theta - y) .^ 2);
endfunction

function J = linearRegressionRegularizedCost(X, y, thetas, r, userId)
  J = linearRegressionCost(X, y, thetas, r, userId);
  theta = thetas(userId, :);
  J += (regularizationFactor() / 2) * (sum(sum(X .^ 2)) - sum(X(1) .^ 2)); 
endfunction

function lambda = regularizationFactor()
  lambda = 5;
endfunction

function J = collaborativeFilteringCost(X, y, thetas, r)
  J = 0;
  [numberOfMovies, numberOfUsers] = size(y);
  for i = 1:numberOfUsers
    J += linearRegressionRegularizedCost(X, y, thetas, r, i);
  endfor
endfunction

X = [2 ; 3 ; 4 ; 0];
y = [4 5 0 ; 1 1 1 ; 2 2 4 ; 0 2 0];
theta1 = [1 2];
theta2 = [3 4];
theta3 = [0 -2];
thetas = [theta1 ; theta2 ; theta3];
r = [1 1 0 ; 1 1 1 ; 1 1 1 ; 0 1 0];

disp(collaborativeFilteringCost(X, y, thetas, r));
