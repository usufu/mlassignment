function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1), X];
a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m, 1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
hx = a3;

y_ = zeros(m, num_labels);
for i = 1:m,
    y_(i,y(i)) = 1;
end


J = 0;
for i=1:m,
    J = J + (y_(i,:) * log(hx(i,:)') + ( 1 - y_(i,:)) * log(1 - hx(i,:))');
    %for j=1:num_labels
    %    J = J + (y_(i,j) * log(hx(i,j)) + ( 1 - y_(i,j)) * log(1 - hx(i,j)));
    %end
end
J = -1 * J / m;

rTheta1 = Theta1;
rTheta1(:,1) = 0;

rTheta2 = Theta2;
rTheta2(:,1) = 0;

regtheta1 = sum(sum(rTheta1.^2));
regtheta2 = sum(sum(rTheta2.^2));


reg = (lambda / (2*m)) * (regtheta1 + regtheta2);

J = J + reg;

v1 = 0;
v2 = 0;

z2 = [ones(m, 1), z2];

if (true)
for i=1:m,
    a1i = X(i,:)'; %401,1
    z2i = Theta1 * a1i; % 25, 401; 401,1 -> 25, 1
    a2i = sigmoid(z2i); % 25, 1

    a2i = [1;a2i]; %26, 1
    z3i = Theta2 * a2i; % 10, 26; 26, 1 -> 10, 1
    a3i = sigmoid(z3i); % 10, 1

    d3i = a3i - y_(i, :)';  %10, 1
    d2i = (Theta2' * d3i) .* sigmoidGradient([1;z2i]);
    d2i = d2i(2:end);

    v1 = v1 + d2i * a1i';
    v2 = v2 + d3i * a2i';

end
else
for i=1:m,
    d3 = a3(i, :) - y_(i, :); % 1, 10
    d2 = (d3 * Theta2) .* sigmoidGradient(z2(i, :));
    d2 = d2(2:end);
    v1 = v1 + d2' * a1(i, :); %25, 1; 1, 401
    v2 = v2 + d3' * a2(i, :); %10, 1; 26, 1
end
endif


reg_grad1 = rTheta1 * (lambda / m);
reg_grad2 = rTheta2 * (lambda / m);


Theta1_grad = v1 / m + reg_grad1;
Theta2_grad = v2 / m + reg_grad2;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
