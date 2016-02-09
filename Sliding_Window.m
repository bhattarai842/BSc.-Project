function Sliding_Window()
image_path=input('\n\nEnter the path and file name for input image\n','s');%%location of traget vector
input_image=imread(image_path);
input_image=imresize(input_image,[300 300]);

[x_size y_size]=size(input_image);
for y_cod=1:y_size-20
	for x_cod=1:x_size-20	
		window=input_image(x_cod:x_cod+20,y_cod+20);
	end
end
imshow(input_image)
rectangle('Position',[110,85 20 20], 'LineWidth',1, 'EdgeColor','b')
end
