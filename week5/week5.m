clc;
clear;

% Gradiant Approximater (Gradiant Checker)
function gradent = approximateGradient(theta, cost) 
  n = length(theta);
  gradient = zeros(n, 1);
  EPSILON = 1e-4;
  for i = 1:n
    thetaPlus = theta;
    thetaPlus(i) += EPSILON;
    thetaMinus = theta;
    thetaMinus(i) -= EPSILON;
    gradent(i) = (cost(thetaPlus) - cost(thetaMinus)) / (2 * EPSILON);
  endfor
endfunction

function J = costFunction(theta)
  J = 100 * rand(1, 1);  
endfunction

hypothesis = [0 ; 1 ; 2];
disp(approximateGradient(hypothesis, @costFunction));
