function centroids = kMeansInitCentroids(X, K)
  %KMEANSINITCENTROIDS This function initializes K centroids that are to be 
  %used in K-Means on the dataset X
  %   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
  %   used with the K-Means on the dataset X

  m = size(X, 1);
  % randomly generate K clusters and positions
  randomizedDataSet = randperm(m);
  centroids = X(randomizedDataSet(1:K), :);
end
