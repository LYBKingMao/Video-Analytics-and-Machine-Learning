%This is a single scale sliding window script
clear all
close all
load detectorModel;
I=imread('im3.jpg');
I=histeq(I);
ScaleWidth=18;
ScaleHeight=27;
[m,n]=size(I)
DestinationM=m-ScaleHeight;
DestinationN=n-ScaleWidth;
vector=[];
for i=1:5:DestinationM
    for j=1:5:DestinationN
        if i>=DestinationM
            i=DestinationM;
        end
        if j>=DestinationN
            j=DestinationN;
        end
        CurrentFrame=I(i:i+ScaleHeight-1,j:j+ScaleWidth-1,:);
        hog=hog_feature_vector(CurrentFrame);
        vector=[vector;hog];
    end
end
[eigenVectors, eigenvalues, meanX, Xpca]=PrincipalComponentAnalysis(vector,20);
figure
imshow(I),hold on
for i=1:size(Xpca,1)
     [prediction,distance(i,:)]=SVMTesting(Xpca(i,:),modelSVM);
    if prediction==1
        locM=fix(i/((n-18)/5))*5;   %DestinationN should divide step length, and multiply step length
        locN=mod(i,((n-18)/5))*5;
        Objects(i,:)=[locN,locM,ScaleWidth,ScaleHeight,distance(i,:)];
        %             stop here you can get original amount of detected Objects
            rectangle('Position',[locN,locM,ScaleWidth,ScaleHeight]);
        elseif prediction==-1
            distance(i,:)=0;
        end
end
hold off
Objects(all(Objects==0,2),:)=[];
Objects=NMS(Objects,100)
figure 
imshow(I),hold on
for i=1:size(Objects,1)
    rectangle('Position',[Objects(i,1),Objects(i,2),ScaleWidth,ScaleHeight])
end
hold off