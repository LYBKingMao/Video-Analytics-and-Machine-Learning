[train_image,train_label]=loadFaceImages('face_train.cdataset');
[test_image,test_label]=loadFaceImages('face_test.cdataset');
[m,n]=size(train_image);
[x,y]=size(test_image);
for i=1:m
    image(i,:)=train_image(i,:);
    label(i,:)=train_label(i,:);
end
for i=1:x
    image(m+i,:)=test_image(i,:);
    label(m+i,:)=test_label(i,:);
end
face=find(label==1);
non_face=find(label==-1);
image=[image(face,:);image(non_face,:)];
label=[label(face,:);label(non_face,:)];
count=1;
for i=1:2:length(image)
    train_image(count,:)=image(i,:);
    train_label(count,:)=label(i,:);
    count=count+1;
end
[eigenVectors, eigenvalues, meanX, Xpca]=PrincipalComponentAnalysis(train_image,20);
modelSVM=SVMtraining(Xpca,train_label);
count=1;
for i=2:2:length(image)
    test_image(count,:)=image(i,:);
    test_label(count,:)=label(i,:);
    count=count+1;
end
for i=1:size(test_image,1)
    x=(test_image(i,:)-meanX)*eigenVectors;
    classificationResult(i,1) = SVMTesting(x, modelSVM);
end
comparison = (test_label==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)