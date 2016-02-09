function [features]=PCA_cal()
%This function is intented to calculate the PCA which is used for facial feature extraction
%
%features= arranged eigen vectors according to eigen value from maximum to minimum

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


img=imread('Images/72.jpg');%importing the image from the folder
img=imresize(img,[20 20]);%resize the image 
img_matrix=rgb2gray(img);%conversion of rgb image to gray
img_matrix=histeq(img_matrix);%normalizing the brightness and contrast of image

%%conversion of uint8 cast to double because in matlab multiplication does not valid unit8 cast
img_matrix=double(img_matrix);
m=mean(img_matrix);%calulation of mean 
sum=0;

%%mean fuction gives the mean of their respective columns so we need to add all the columns mean and divide by no of columns to find the total mean of given image color %%intensity
for i=1:size(m,2)
	sum=sum+m(1,i);
end
pixel_mean=sum/size(m,2);

%%%Centering the image pixels
img_matrix=img_matrix-pixel_mean;

%%Calculation of covariance
co_matrix=img_matrix*img_matrix';


%%calculation of eigen value and eigen vectors of given co_matrix

%[V,D] = EIG(X) produces a diagonal matrix D of eigenvalues and a
% full matrix V whose columns are the corresponding eigenvectors so
% that X*V = V*D

[img_eigen img_eigen_v]=eig(co_matrix);

features=sort(diag(img_eigen));

%%Sorting the eigen vectors form its eigen usind subroutine sortem(V,D)

[new_eigen new_eigen_v]=sortem(img_eigen,img_eigen_v);

%%Unrolling the 2D eigenvector to 1D feature vector which will be input to our Neural Network Input Layer
[irow icol] = size(new_eigen);
%features = reshape(new_eigen,irow*icol,1); 

%%end of function
end
