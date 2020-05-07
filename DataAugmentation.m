for i=1:20
    filename=sprintf('./images/face/%d.png',i);
    image=double(imread(filename))
    image=image+wgn(size(image,1),size(image,2),20);
    filename=sprintf('./Augmentation/face/%d.png',i);
    imwrite(image,filename);
end
for i=1:10
    filename=sprintf('./images/non-face/%d.png',i);
    image=double(imread(filename))
    image=flip(image,1);
    filename=sprintf('./Augmentation/non-face/%d.png',i);
    imwrite(image,filename);
end
for i=11:15
    filename=sprintf('./images/non-face/%d.png',i);
    image=double(imread(filename))
    image=flip(image,2);
    filename=sprintf('./Augmentation/non-face/%d.png',i);
    imwrite(image,filename);
end
for i=26:30
    filename=sprintf('./images/non-face/%d.png',i);
    image=double(imread(filename))
    image=flip(image,3);
    filename=sprintf('./Augmentation/non-face/%d.png',i-10);
    imwrite(image,filename);
end