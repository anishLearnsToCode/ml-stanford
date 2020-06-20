function X = recoverData(Z, U, K)
  %RECOVERDATA Recovers an approximation of the original data when using the 
  %projected data
  %   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
  %   original data that has been reduced to K dimensions. It returns the
  %   approximate reconstruction in X_rec.

  uReduce = U(:, 1:K);
  X = Z * uReduce';
end
