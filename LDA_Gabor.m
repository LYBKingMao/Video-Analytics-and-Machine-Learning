[train_image,train_label]= loadFaceImages('face_train.cdataset');
face=find(train_label==1);
non_face=find(train_label==-1);
image=[train_image(face,:);train_image(non_face,:)];
label=[train_label(face,:);train_label(non_face,:)];
vector=[];
for j=1:size(train_image,1)
        Ti=reshape(train_image(j,:),27,18)
        gabor=gabor_feature_vector(Ti);
        vector=[vector;gabor];
    end
[eigenVectors, eigenvalues, meanX, Xlda]=LDA(label,[],vector);    
modelSVM = SVMtraining(Xlda, label);
[images,labels]= loadFaceImages('face_test.cdataset');
face=find(labels==1);
non_face=find(labels==-1);
images=[images(face,:);images(non_face,:)];
labels=[labels(face,:);labels(non_face,:)];
for i=1:size(images,1)
    Ti=reshape(images(i,:),27,18)
    gabor=gabor_feature_vector(Ti);
    x=(gabor-meanX)*eigenVectors;
    classificationResult(i,1) = SVMTesting(x, modelSVM);
    
end
comparison = (labels==classificationResult);
Accuracy = sum(comparison)/length(comparison)

