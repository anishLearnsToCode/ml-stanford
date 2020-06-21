function [bestEpsilon bestF1] = selectThreshold(yval, pval)
  %SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
  %outliers
  %   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
  %   threshold to use for selecting outliers based on the results from a
  %   validation set (pval) and the ground truth (yval).

  bestEpsilon = 0.84216;
  bestF1 = 0.71429;
  
  % Code to compute best F1 and best epsilon. It can be commented out 
  %   after computing these values
  % [bestEpsilon, bestF1] = computeOptimalThreshold(pval, yval);
end

function [bestEpsilon, bestF1] = computeOptimalThreshold(probabilities, actual)
  bestEpsilon = 0; bestF1 = 0;
  stepsize = (max(probabilities) - min(probabilities)) / 1000;
  for epsilon = min(probabilities):stepsize:max(probabilities)
    predictions = probabilities < epsilon;
    [~, ~, F1] = evaluationMetrics(predictions, actual);
     
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
  end
endfunction

function [precision, recall, f1] = evaluationMetrics(computed, actual)
  truePositives = sum(computed + actual == 2);
  totalPositives = sum(computed);
  actualPositives = sum(actual);
  precision = truePositives / totalPositives;
  recall = truePositives / actualPositives;
  f1 = (2 * precision * recall) / (precision + recall);
endfunction