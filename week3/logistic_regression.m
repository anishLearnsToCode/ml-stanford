clc;
clear;

function [value, gradient] = costFunction(theta) 
  value = (theta(1) - 5)^2 + (theta(2) - 10)^2;
  gradient = zeros(2, 1);
  gradient(1) =   2 * (theta(1) - 5);
  gradient(2) =   2 * (theta(2) - 10);
endfunction


options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2, 1);
[theta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
disp(theta);
disp(functionVal);
disp(exitFlag);
