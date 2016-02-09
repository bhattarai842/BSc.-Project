function[J grad]=ANN(parameter_weight,input_layer_size,hidden_layer_size,labels,features, y, lambda,m)
%%ANN used in this code is gradient decent. The purpose of this code is to feed the features vector of image from PCA_cal() to find wheather the image is 
%face or not. In this network we will have only one hidden layer. 
% features is feature matrix from PCA_cal
% y is target output

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=size(features, 1);
%%Reshaping the value of theta1 and theta2
Theta1 = reshape(parameter_weight(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(parameter_weight((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 labels, (hidden_layer_size + 1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%feed forward
input_ANN = [ones(m, 1) features];%%adding bias to ANN
a1=input_ANN;
z2=Theta1*input_ANN';             %%Multiplication and addition of weights and input in first layer
a2=sigmoid(z2);                   %%Out put after the sigmoid function from hidden layer
a2=a2';
a2=[ones(size(a2,1),1) a2];       %%Adding bias to hidden layer
z3=a2*Theta2';
a3=sigmoid(z3);                   %%feedforward final output

%%%Cost of forward propagation

J=1/m*(sum(sum((-y.*log(a3))-((1-y).*log(1-a3)))));


[t1n t1m]=size(Theta1);
[t2n t2m]=size(Theta2);


st1=sum(sum(Theta1(:,2:t1m).^2));
st2=sum(sum(Theta2(:,2:t2m).^2));
reg=(st1+st2)*lambda/(2*m);

J=J+reg;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%backward propagation of ANN


Delta1=zeros(size(Theta1));
Delta2=zeros(size(Theta2));


delta_3=(a3-y);
delta_2=(delta_3*Theta2(:,2:end)).*sigmoid(z2)';
Delta1=delta_2'*a1;
Delta2=delta_3'*a2;
D1=1/m*Delta1;
D2=1/m*Delta2;

%%Calculating partial derivative for optimizing weights of nerual network
%%Computation for input to hidden layer

for i=1:size(D1,1)
	for j=1:size(D1,2)

	if j==1
		D1(i,j)=D1(i,j);
	else
		D1(i,j)=D1(i,j)+(Theta1(i,j)*lambda)/m;
	end
end
end

%%Comutation for hidden to output layer

for i=1:size(Delta2,1)
	for j=1:size(Delta2,2)

	if j==1
		D2(i,j)=D2(i,j);
	else
		D2(i,j)=D2(i,j)+(Theta2(i,j)*lambda)/m;
	end
end
end

%% Unroll gradients
grad = [D1(:) ; D2(:)];


end
