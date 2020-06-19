function [C, sigma] = dataset3Params(X, y, Xval, yval)
  %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
  %where you select the optimal (C, sigma) learning parameters to use for SVM
  %with RBF kernel
  %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
  %   sigma. You should complete this function to return the optimal C and 
  %   sigma based on a cross-validation set.
  %
  
  C = 1;
  sigma = 0.1;
 
  % Code to compute the parameters with minimum cost
  % [C, sigma] = computeMinCostParameters()
  
  function [C, sig] = computeMinCostParameters()
    results = eye(64,3);
    errorRow = 0;

    for C_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
        for sigma_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
            errorRow = errorRow + 1;
            model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
            predictions = svmPredict(model, Xval);
            prediction_error = mean(double(predictions ~= yval));
            results(errorRow,:) = [C_test, sigma_test, prediction_error];     
        end
    end

    sorted_results = sortrows(results, 3); % sort matrix by column #3, the error, ascending

    C = sorted_results(1,1);
    sig = sorted_results(1,2);
    disp('C');      disp(C);
    disp('sigma');  disp(sig);  
  endfunction
end
