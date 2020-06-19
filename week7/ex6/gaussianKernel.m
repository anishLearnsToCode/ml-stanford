function sim = gaussianKernel(x1, x2, sigma)
  %RBFKERNEL returns a radial basis function kernel between x1 and x2
  %   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
  %   and returns the value in sim

  % Ensure that x1 and x2 are column vectors
  x1 = x1(:); x2 = x2(:);

  difference = x1 - x2;
  euclideanDistanceSquared = sum(difference .^ 2);
  variance = sigma ^ 2;
  sim = exp(- euclideanDistanceSquared / (2 * variance));
end
