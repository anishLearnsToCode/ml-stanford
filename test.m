clc;

function result = square(number)
  result = number * number;
end

function [sq, cube] = squareAndCube(number)
  sq = square(number);
  cube = sq * number;
end  

function cost = costFunction(x, hypothesis, result)
  trainingExamles = size(x)(1);
  predictions = x * hypothesis;
  squareErrors = (predictions - result) .^ 2;
  cost = 0.5 / trainingExamles * sum(squareErrors);
end


vector = [1 ; 2; 3; 4; 5];

for i = 1:length(vector),
  vector(i) = 2^i;
end

disp(vector);
disp(square(3))
[sq, c] = squareAndCube(10);
disp(sq)

x = [1 1 ; 1 2 ; 1 3];
hypothesis = [100 ; 0];
y = [1 ; 2; 3];
disp(costFunction(x, hypothesis, y));


disp(computeCost(x, y, hypothesis));
gradientDescent()