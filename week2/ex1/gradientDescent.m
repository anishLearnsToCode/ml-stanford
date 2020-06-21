grfunction [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
  %   taking num_iters gradient steps with learning rate alpha

  % Initialize some useful values
  trainingExamples = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  features = length(theta) - 1;

  for iter = 1:num_iters
     theta = theta - alpha * (1/trainingExamples) * (((X*theta) - y)' * X)'; % Vectorized  
     J_history(iter) = computeCost(X, y, theta);
  end
end
