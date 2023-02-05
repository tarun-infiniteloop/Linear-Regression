function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

J_history(1) = computeCost(X,y,theta);
delta = zeros(2,1);

for iter = 1:num_iters
    old_theta = theta;

        for i = 1:2
            delta(i) = sum(((X*old_theta)-y).*(X(:,i)));
        end

        theta = old_theta - (alpha/m)*delta;
   
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
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    %J_history(iter) = computeCost(X, y, theta);


