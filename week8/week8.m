clc;
clear;

% Function used to display elbow plot
function costs = clusterVariationCosts(X)
  K = size(X, 1);
  for i = 1:K
    [~, minCost] = multipleKMeans(X, i, 5);
    costs(i) =  minCost; 
  endfor
endfunction

function [kMeans, minCost] = multipleKMeans(X, K, iters)
  n = size(X, 2);
  kMeans = zeros(iters, K, n);
  cost = zeros(iters, 1);
  for i = 1:iters
    [clusters, J] = kMeansClustering(X, K);
    kMeans(i, :, :) = clusters;
    cost(i) = J;
  endfor
  [minCost, index] = min(cost);
  kMeans = reshape(kMeans(index, :, :), K, n);
endfunction

function [clusters, minCost] = kMeansClustering(X, K)
  m = size(X, 1);
  n = size(X, 2);
  
  % randomly generate K clusters and positions
  randomizedDataSet = randperm(m);
  clusters = X(randomizedDataSet(1:K), :);
  
  % Run the K-means clustering algorithm
  for iter = 1:10
    pointGroup = distanceFromClusterPoints(X, K, clusters);
    [minDistace, index] = min(pointGroup, [], 2);
    cost = (1 / m) * sum(minDistace);
    clusters = centroids(X, K, index, clusters);
  endfor
  minCost = cost;
endfunction

function mat = distanceFromClusterPoints(X, K, clusters)
  m = size(X, 1);
  mat = zeros(m, K);
  for i = 1:K
    mat(:, i) = euclideanDistanceSquare(X, clusters(i, :));
  endfor
endfunction

function c = centroids(X, K, clusterData, clusters)
  n = size(X, 2);
  m = size(X, 1);
  c = zeros(K, n);
  frequency = zeros(K, 1);
  for i = 1:m
    frequency(clusterData(i))++;
    c(clusterData(i), :) += X(i, :);
  endfor
  mask = c == 0;
  frequency = maskZeroAsOne(frequency);
  c = c ./ frequency;
  c += mask .*  clusters;
endfunction

function mat = maskZeroAsOne(mat)
  mask = mat == 0;
  mat += mask;
endfunction

function d = euclideanDistanceSquare(X, cluster)
  trainingDataSize = size(X, 1); 
  similarityMatrix = repelem(cluster, trainingDataSize, 1);
  difference = X - similarityMatrix;
  d = sum(difference .^ 2, 2);
endfunction

labels = 5;
data = [-10 1 ; 11 2 ; 45 3 ; 4 4 ; 7 5 ; 100 100 ; 5 -8 ; -89 23];
% disp('K means clustering - single');
% disp(kMeansClustering(data, labels));

disp('multiple k means');
[kMeans, minCost] = multipleKMeans(data, labels, 10);
disp('min cost'); disp(minCost);
disp('k means'); disp(kMeans);

% Elbow plot
plot(clusterVariationCosts(data));
