function [final_output]=Detection(input_image,final_Theta1,final_Theta2);

	img=imread(input_image);
	img=imresize(img,[20 20]);%resize the image 
	img_matrix=rgb2gray(img);%conversion of rgb image to gray
	img_matrix=histeq(img_matrix);%normalizing the brightness and contrast of image

	%%conversion of uint8 cast to double because in matlab multiplication does not valid unit8 cast
	img_matrix=double(img_matrix);
	mean_value=mean(img_matrix);%calulation of mean 
	sum=0;

	%%mean fuction gives the mean of their respective columns so we need to add all the columns mean and divide by no of columns to find the total mean of given 		image color %%intensity
	for i=1:size(mean_value,2)
		sum=sum+mean_value(1,i);
	end
	pixel_mean=sum/size(mean_value,2);

	%%%Centering the image pixels
	img_matrix=img_matrix-pixel_mean;


	%%calculation of eigen value and eigen vectors of given img_matrix using SVD

	%[U,S,V] = SVD(X) produces a diagonal matrix S, of the same 
	%dimension as X and with nonnegative diagonal elements in
	%decreasing order, and unitary matrices U and V so that
	%X = U*S*V'.


	[U,S,V]=svd(img_matrix);%%Using SVD subroutine of matlab

	%%Unrolling the 2D eigenvector to 1D feature vector which will be input to our Neural Network Input Layer
	[irow icol] = size(U);
	old_features = reshape(U,irow*icol,1); 
	features=old_features(1:220,:);
	features=features';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%feeding to Neural Network

input_ANN = [1 features];%%adding bias to ANN
a1=input_ANN;
z2=final_Theta1*input_ANN';             %%Multiplication and addition of weights and input in first layer
a2=sigmoid(z2);                   %%Out put after the sigmoid function from hidden layer
a2=a2';
a2=[ones(size(a2,1),1) a2];       %%Adding bias to hidden layer
z3=a2*final_Theta2';
a3=sigmoid(z3);                   %%feedforward final output
final_output=a3;

end
