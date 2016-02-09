function Weights = Theta_Weights(L_in, L_out)
%Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = Theta_WEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first row of W handles the "bias" terms
%

% You need to return the following variables correctly 
Weights = zeros(L_out, 1 + L_in);
epsilon_init=0.12;
Weights=rand(L_out,1+L_in)*2*epsilon_init-epsilon_init;

% =========================================================================

end
