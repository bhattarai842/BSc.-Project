function mainfunction()

%%Main function of FACE DETECTION AND RECOGNITION SYSTEM
fprintf(2,'    ***********************************************************    \n');
fprintf(2,'    ***********FACE DETECTION AND RECOGNITION******************    \n');
fprintf(2,'    ***********************************************************    \n\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Extracting features of images
[features]=SVD_cal();

y_path=input('Enter the path and file name for target value vector (y)\n','s');%%location of traget vector
y=load(y_path);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Initializing weight of Neural Network
input_layer_size  = 220;  % features form PCA
hidden_layer_size = 170;   % 25 hidden units
labels=1;%%number of output
lambda=1;%%Regularization cofficient

%%Initializing the weights for Theta1 and Theta2

Theta1 = Theta_Weights(input_layer_size, hidden_layer_size);
Theta2 = Theta_Weights(hidden_layer_size,labels);

initial_weight_parameter=[Theta1(:) ; Theta2(:)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%weight optimization using cojugate garadient descent

options = optimset('MaxIter',50);
cost_ANN = @(t) ANN(t,input_layer_size,hidden_layer_size,labels, features, y, lambda);

[final_weights_parm] = fmincg(cost_ANN,initial_weight_parameter,options);%%inbuit function for optimization in matlab

%%Reshaping the value of theta1 and theta2
final_Theta1 = reshape(final_weights_parm(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

final_Theta2 = reshape(final_weights_parm((1 + (hidden_layer_size * (input_layer_size + 1))):end),labels, (hidden_layer_size + 1));

fprintf(2,'\n\n VALUE OF WEIGHT OPTIMIZED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Detecing the face and non face

input_image=input('\n\nEnter the path and file name for input image\n','s');%%location of traget vector
checking_value=Detection(input_image,final_Theta1,final_Theta2)%%Calling function to check the input image is face or not

if(checking_value>0.4)
	fprintf(2,'\n\nGiven Image is Human face\n');
else
	fprintf(2,'\n\n Sorry!! Image is not Human face\n');
end


end
