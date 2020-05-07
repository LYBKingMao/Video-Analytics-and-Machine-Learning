i1=imread('im3.jpg');
figure
imshow(i1)
figure
imshow(histeq(i1))
figure
imshow(adapthisteq(i1))