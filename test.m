clc;
clear;

function value = sigmoid(matrix)
  value = 1./ (1 + exp(-matrix));
endfunction

X = [1 1 ; 1 2 ; 1 3];
all_theta = [0 0 ; 1 1 ; 2 2 ; 3 3];

hypotheses = X * all_theta';
disp(hypotheses);
probabilities = sigmoid(hypotheses);
disp(probabilities);

[maxProbabilities index] = max(probabilities, [], 2);
disp(maxProbabilities);
disp(index);
