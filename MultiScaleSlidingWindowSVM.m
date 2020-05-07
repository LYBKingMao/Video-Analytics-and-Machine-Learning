clear all
close all
load detectorModel;
I=imread('im1.jpg');
I=adapthisteq(I);
[m,n]=size(I)
StartWidth=18;
StartHeight=27;
Objects=[];
distance=[];
count=0;

%Set 5 iteration as default, each iteration make sliding window size multiply 1.1
for i=1:5
    DestinationM=m-floor(StartHeight*power(1.1,i-1));
    DestinationN=n-floor(StartWidth*power(1.1,i-1));
    vector=[];
    Scale=[];
    for j=1:5:DestinationM
        for k=1:5:DestinationN
            if j>=DestinationM
                j=DestinationM;
            end
            if k>=DestinationN
                k=DestinationN;
            end
            
            %Cut current frame, calculate features, store to vector
            CurrentFrame=I(j:j+floor(StartHeight*power(1.1,i-1)),k:k+floor(StartWidth*power(1.1,i-1)),:);
            hog=hog_feature_vector(CurrentFrame);
            vector=[vector;hog];
        end
    end
    
    %PCA applied and parameter set
    [eigenVectors, eigenvalues, meanX, Xpca]=PrincipalComponentAnalysis(vector,20);
    
    %input features to classifier
    for j=1:size(Xpca,1)
        count=count+1;
        [prediction,distance(count,:)]=SVMTesting(Xpca(j,:),modelSVM);
        if prediction==1
            locM=fix(j/((n-StartHeight*power(1.1,i-1))/5))*5;   %DestinationN should divide step length, and multiply step length
            locN=mod(j,(n-StartWidth*power(1.1,i-1))/5)*5;
            Objects(count,:)=[locN,locM,floor(StartWidth*power(1.1,i-1)),floor(StartHeight*power(1.1,i-1)),distance(count,:)];
%             stop here you can get original amount of detected Objects
%             rectangle('Position',[locN,locM,floor(StartWidth*power(1.1,i-1)),floor(StartHeight*power(1.1,i-1))]);
        elseif prediction==-1
            distance(count,:)=0;
        end
    end
end
Objects(all(Objects==0,2),:)=[];

%Ensure sliding windows not too general(big)
pos=find(Objects(:,5)>4.0)
Objects(pos,:)=[];

%NMS applied
Objects=NMS(Objects,200);

%draw rectangle
figure 
imshow(I),hold on
for j=1:size(Objects,1)
    rectangle('Position',[Objects(j,1),Objects(j,2),floor(StartWidth*power(1.1,i-1)),floor(StartHeight*power(1.1,i-1))]);
end
hold off