function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X,2);
J_history(1) = computeCost(X,y,theta);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
  
    theta = theta - (1/m)*(X' * (X*theta - y));




    % ============================================================

    % Save the cost J in every iteration    
     J = computeCost(X,y,theta);

    if (J > J_history(iter))
       break;
    else 
       if (iter+1 > num_iters)
           break
       else
           J_history(iter+1) = J;
       end
       continue;
    end
end

end
