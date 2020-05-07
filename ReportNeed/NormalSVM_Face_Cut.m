[train_image,train_label]= loadFaceImages('face_train.cdataset');
face=find(train_label==1);
non_face=find(train_label==-1);
image=[train_image(face,:);train_image(non_face,:)];
label=[train_label(face,:);train_label(non_face,:)];
modelSVM=SVMtraining(image,label);
[test_image,test_label]=loadFaceImages('face_test.cdataset');
face=find(test_label==1);
non_face=find(test_label==-1);
image=[test_image(face,:);test_image(non_face,:)];
label=[test_label(face,:);test_label(non_face,:)];
for i=1:size(image,1)
    
    testnumber= image(i,:);
    
    classificationResult(i,1) = SVMTesting(testnumber, modelSVM);
    
end
comparison = (label==classificationResult);
Accuracy = sum(comparison)/length(comparison)