function idx = findClosestCentroids(X, centroids)
  %FINDCLOSESTCENTROIDS computes the centroid memberships for every example
  %   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
  %   in idx for a dataset X where each row is a single example. idx = m x 1 
  %   vector of centroid assignments (i.e. each entry in range [1..K])

  K = size(centroids, 1);
  pointGroup = distanceFromClusterPoints(X, K, centroids);
  [~, idx] = min(pointGroup, [], 2);
  
  function mat = distanceFromClusterPoints(X, K, clusters)
    m = size(X, 1);
    mat = zeros(m, K);
    for i = 1:K
      mat(:, i) = euclideanDistanceSquare(X, clusters(i, :));
    endfor
  endfunction
  
  function d = euclideanDistanceSquare(X, cluster)
    trainingDataSize = size(X, 1); 
    similarityMatrix = repelem(cluster, trainingDataSize, 1);
    difference = X - similarityMatrix;
    d = sum(difference .^ 2, 2);
  endfunction
end
