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
vector=[];
[m,n]=size(image)
indices=crossvalind('Kfold',m,10);
record=[];
sumacc=0;
for i=1:10
    test_index=(indices==i);
    train_index=~test_index;
    train_image=image(train_index,:);
    train_label=label(train_index,:);
    test_image=image(test_index,:);
    test_label=label(test_index,:);
    for j=1:size(train_image,1)
        Ti=reshape(train_image(j,:),27,18)
        hog=hog_feature_vector(Ti);
        vector=[vector;hog];
    end
    [eigenVectors, eigenvalues, meanX, Xpca]=PrincipalComponentAnalysis(vector,21);
    modelSVM=SVMtraining(Xpca,train_label);
    for k=1:size(test_image,1)
        Ti=reshape(test_image(k,:),27,18)
        hog=hog_feature_vector(Ti);
        x=(hog-meanX)*eigenVectors;
        classificationResult(k,:)=SVMTesting(x,modelSVM);
    end
    comparison=(test_label==classificationResult);
    Accuracy=sum(comparison)/length(comparison);
    record=[record;Accuracy];
    sumacc=sumacc+Accuracy;
    vector=[];
end
meanacc=sumacc/10