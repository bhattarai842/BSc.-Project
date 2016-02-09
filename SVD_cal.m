function [features]=SVD_cal()
%This function is intented to calculate the PCA which is used for facial feature extraction
%
%features= arranged eigen vectors according to eigen value from maximum to minimum

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yes_no=input('Press Y if you have training matrix in text file else N\n','s');

if(yes_no=='y'||yes_no=='Y')
	file_location=input('Enter the location of the feature matrix text \n','s');
	features=load('features.txt');
else
	fprintf('Images must be in numeric order from 1 to infinty and in .jpg format\n\n\n');
	TrainDatabasePath=input('Enter the path of the folder of training images\n','s');%%location of traning images
	TrainFiles = dir(TrainDatabasePath);
	Train_Number = 0;

	for i = 1:size(TrainFiles,1)
    		if not(strcmp(TrainFiles(i).name,'.')|strcmp(TrainFiles(i).name,'..')|strcmp(TrainFiles(i).name,'Thumbs.db'))
        		Train_Number = Train_Number + 1; % Number of all images in the training database
    		end
	end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	features= [];
	fprintf(2,'\n\tCalculating features of images...........\n\n\n');
	for i = 1 : Train_Number
        	str = int2str(i);
        	str = strcat('/',str,'.jpg');
        	str = strcat(TrainDatabasePath,str);

		img=imread(str);
		img=imresize(img,[20 20]);%resize the image 
		img_matrix=rgb2gray(img);%conversion of rgb image to gray
		img_matrix=histeq(img_matrix);%normalizing the brightness and contrast of image

		%%conversion of uint8 cast to double because in matlab multiplication does not valid unit8 cast(Casting)
		img_matrix=double(img_matrix);
		m=mean(img_matrix);%calulation of mean 
		sum=0;

		%%mean fuction gives the mean of their respective columns so we need to add all the columns mean and divide by no of columns to find the total mean of
          	%%  givenimage color intensity
		for i=1:size(m,2)
			sum=sum+m(1,i);
		end
		pixel_mean=sum/size(m,2);

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
		old_features = reshape(U,irow*icol,1); %%Converting 2D matrix to 1D matrix
		old_features=old_features(1:220,:);
		features=[features old_features];
	end%%for loop end
features=features';

fid = fopen('features.txt','wt');
for i = 1:size(features,1)
    fprintf(fid,'%g\t',features(i,:));
    fprintf(fid,'\n');
end
fclose(fid);


end%% end of if else
%%end of function
end
