clc;
clear;
close all;

% @return J:scalar cost of the linear regression given theta 
% @param movieData: (n_m),(n) feature vector describing every movie
% @param userRatingsData: (n_m),(n_u) rating of every movie by each user that has 
%   rated the move
% @param userId:scalar - represents the user numebr that we are calculating the 
%   cost for given our cuirrent hypothesis
% @param ratedMovies: (n_m),(n_u): matrix containing 1/0 1 specifies user has   
%   rated movie and 0 not rated movie
% @param theta:(n+1),1 hypothesis for the given user
% (n_m): Number of movies (n_u): Number of users
function [J, m] = linearRegressionCost(movieData, userRatingsData, userId, ratedMovies,
  theta)
  %indices of movies rated by given user
  movies = find(ratedMovies(:, userId) == 1);
  
  % Getting data vectors of rated movies 
  movieData = movieData(movies, :);
  
  [m n] = size(movieData);
  
  % Adding parameter x_0 in all movie data vectors
  X = [ones(m, 1) movieData];
  
  % Getting the rating of every movie that this user has rated
  y = userRatingsData(movies, userId);
  
  J = (1 / 2) * sum((X * theta - y) .^ 2);
endfunction

function J = linearRegressionRegularizedCost(movieData, userRatingsData, userId,
  ratedMovies, theta)
  
  [J, m] = linearRegressionCost(movieData, userRatingsData, userId, ratedMovies, theta);
  J += (regularizationParam() / 2) * (sum(theta .^ 2) - theta(1) ^ 2);
endfunction

function J = linearRegressionCostCombined(X, Y, thetas, r)
  [numberOfMovies, numberOfUsers] = size(Y);
  J = 0;
  for userId = 1:numberOfUsers
    J += linearRegressionRegularizedCost(X, Y, userId, r, thetas(userId, :)');
  endfor  
endfunction

function lambda = regularizationParam()
  lambda = 5;
endfunction

function g = gradient(X, y, thetas)
  [numberOfMovies, features] = size(X);
  [numberOfMovies, numberOfUsers] = size(y);
  X = [ones(numberOfMovies, 1) X];
  g = (X' * (X * thetas' - y))';
  mask = regularizationParam() * ones(numberOfUsers, features + 1);
  mask(:, 1) = 0;
  g += mask .* thetas;
endfunction

function [minCost, thetas, costs] = gradientDescent(X, y, thetas, r, iterations, alpha)
  costs = [];
  for i = 1:iterations
    thetas -= alpha * gradient(X, y, thetas);
    minCost = linearRegressionCostCombined(X, y, thetas, r);
    costs = [costs minCost];
  endfor
endfunction 

x = [1 2 ; 3 4];
r = [0 ; 1 ; 0 ; 0];
userId = 2;
ratings = [0 3 5 ; 5 2 1];
ratedMovies = [0 1 1 ; 1 1 1];
theta1 = [1 ; 2 ; 0];
theta2 = [0 ; 4 ; -3];
theta3 = [10 ; -9 ; 4];
thetas = [theta1' ; theta2' ; theta3'];
% disp(linearRegressionCost(x, ratings, 2, ratedMovies, theta2));
% disp(linearRegressionRegularizedCost(x, ratings, 2, ratedMovies, theta2));
% disp(linearRegressionCostCombined(x, ratings, [theta1' ; theta2' ; theta3'], ratedMovies));
% disp(gradient(x, ratings, thetas));

[minCost, thetas, costs] = gradientDescent(x, ratings, thetas, ratedMovies, 1000, 0.03);
disp('min cost'); disp(minCost);
disp('thetas'); disp(thetas);
plot(costs);
