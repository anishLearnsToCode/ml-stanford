clc;
clear;

y = [1 1 ; 1 2 ; 1 3];
y = y == 2;
disp(y);

function row = labelVector(len, label)
    row = zeros(len, 1);
    row(label) = 1;
 endfunction
 
 disp(labelVector(4, 2)');
 disp(labelVector(3, 1)');
 disp(labelVector(10, 6)');
 