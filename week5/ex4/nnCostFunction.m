function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
 
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  m = size(X, 1);
         
  % You need to return the following variables correctly 
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  X = [ones(m,1) X];
  a1 = X;
  z2 = Theta1 * a1'; 
  a2 = sigmoid(z2);

  a2 = [ones(m,1) a2'];
  z3 = Theta2 * a2';
  a3 = sigmoid(z3); 
  
  % Computing cost for given Theta parameters
  result = resultMatrix(y);
  J = regularizedLogisticCost(result, a3);  
  
  function mat = resultMatrix(dataVector)
    mat = zeros(num_labels, 0);
    for i = 1:length(dataVector)
      mat = [mat labelVector(num_labels, y(i))];  
    endfor
  endfunction
  
  function row = labelVector(len, label)
    row = zeros(len, 1);
    row(label) = 1;
  endfunction
  
  function J = logisticRegressionCost(actual, computed)
    J = - (1 / m) * sum(sum(actual .* log(computed) 
    + (1 - actual) .* log(1 - computed)));
  endfunction
  
  function J = regularizedLogisticCost(actual, computed)
    J = logisticRegressionCost(actual, computed);
    % Note we should not regularize the terms that correspond to the bias. 
    [theta1NonBias, theta2NonBias] = nonBiasThetas();
    J +=  (lambda / (2 * m)) * (
      elementSum(theta1NonBias .^ 2) + elementSum(theta2NonBias .^ 2)  
    );
  endfunction
  
  function [t1, t2] = nonBiasThetas()
    % For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
    t1 = Theta1(:,2:size(Theta1,2));
    t2 = Theta2(:,2:size(Theta2,2));
  endfunction
  
  function ans = elementSum(matrix)
    ans = sum(sum(matrix));
  endfunction

  % Part 2: Implement the backpropagation algorithm to compute the gradients
  %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  %         Theta2_grad, respectively. After implementing Part 2, you can check
  %         that your implementation is correct by running checkNNGradients
  
  for t = 1:m
    a1 = X(t,:)';
    z2 = Theta1 * a1; 
    a2 = sigmoid(z2); 
      
    a2 = [1 ; a2]; % adding a bias 
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
      
    delta_3 = a3 - result(:,t); 
    z2 = [1; z2]; 
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);
    delta_2 = delta_2(2:end);

    Theta2_grad += delta_3 * a2'; 
    Theta1_grad += delta_2 * a1'; 
  end;
  
  Theta2_grad /= m; 
  Theta1_grad /= m; 

  % Part 3: Implement regularization with the cost function and gradients.

  regularizationMaskTheta1 = (lambda / m) * ones(size(Theta1));
  regularizationMaskTheta2 = (lambda / m) * ones(size(Theta2));
  regularizationMaskTheta1(:, 1) = zeros(size(Theta1, 1), 1);
  regularizationMaskTheta2(:, 1) = zeros(size(Theta2, 1), 1);
  
  Theta1_grad += regularizationMaskTheta1 .* Theta1;
  Theta2_grad += regularizationMaskTheta2 .* Theta2;

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
