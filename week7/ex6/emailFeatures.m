function x = emailFeatures(word_indices)
  %EMAILFEATURES takes in a word_indices vector and produces a feature vector
  %from the word indices
  %   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
  %   produces a feature vector from the word indices. 

  % Total number of words in the dictionary
  n = 1899;

  % You need to return the following variables correctly.
  x = zeros(n, 1);

  for i = 1:length(word_indices)
    x(word_indices(i)) = 1;  
  endfor
end
