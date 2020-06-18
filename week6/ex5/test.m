p = 3;
X = [1 ; 2 ; 3];
X_poly = X_poly = zeros(numel(X), p);
X_poly(:, 1) = X;

for degree = 2:p
    X_poly(:, degree) = X_poly(:, degree - 1) .* X;
endfor

disp(X_poly);
