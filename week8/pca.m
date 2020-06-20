clc;
clear;
close all;

% Principal Component Analysis (PCA) Algorithm
function x = meanNormalizedVectors(x)
  mu = mean(x);
  maxValues = max(x);
  x -= mu;
  % x ./= maxValues;
endfunction

% @return n,k
function [u, S] = reducedDimensionSpace(X, k)
  m = size(X, 1);
  covMatrix = (1 / m) * X' * X;
  [eigenVectors, S, ~] = svd(covMatrix); % eigenVectors: n,n n=#features(X)
  u = eigenVectors(:, 1:k);
endfunction

function z = dimensionallyReducedData(X, k)
  % X = meanNormalizedVectors(X);
  reductionMatrix = reducedDimensionSpace(X, k);
  z = X * reductionMatrix;
endfunction

% @return x;m,n
% @param z: m,k
function x = dimensionallyReconstructedData(z, reducedDimMask)
  x = z * reducedDimMask';  
endfunction

% @return v:scalar
% @param original:m,n
% @param reconstructed:m,n 
function v = pcaVariance(original, reconstructed)
  difference = original - reconstructed;
  euclidianDistanceSquaredCost = sum(sum(difference .^ 2));
  distanceFromOriginCost = sum(sum(original .^ 2));
  v = euclidianDistanceSquaredCost / distanceFromOriginCost ;
endfunction  

% @return v: scalar (variance)
% @param S:n,n diagonal matrix
% @param k:scalar #reducedDimensions
function v = variancePcaUsingCovMat(S, k)
  reducedDimensionSum = sum(sum(S(1:k, :)));
  completeDimensionSum = sum(sum(S));
  v = 1 - (reducedDimensionSum / completeDimensionSum);  
endfunction

function k = minDimensionalityReduction(x, threshold)
  n = size(x, 2);
  [~, S] = reducedDimensionSpace(x, 1);
  for i = 1:n
    variance = variancePcaUsingCovMat(S, i);
    if variance < threshold
      k = i;
      break;
    endif
  endfor
endfunction

x = [1 2 3 ; 4 5 6 ; 10 10 0 ; 100 -90 34];
x2 = [1 2 ; 4 5];
% disp(meanNormalizedVectors(x));
% disp('original data'); disp(x);
% disp('dimensionally reduced data'); disp(dimensionallyReducedData(x, 3));

u = reducedDimensionSpace(x, 3);
z = dimensionallyReducedData(x, 3);
r = dimensionallyReconstructedData(z, u);
% disp('reconstruction of dimensionally reduced data'); disp(r);

% variance in PCA algo
disp('min dimension data can be reduced to'); disp(minDimensionalityReduction(x, 0.1));
