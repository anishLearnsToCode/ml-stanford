clc;
clear;

A = rand(5, 3);
B = rand(3, 5);
C = A * B;
disp('c'); disp(C);
R = rand(5, 5) >= 0.5 ;
disp('r'); disp(R);

disp(sum(sum(C(R == 1))));
% C = (A * B) * R;

C = A(R == 1) * B(R == 1);
disp(sum(sum(C)));