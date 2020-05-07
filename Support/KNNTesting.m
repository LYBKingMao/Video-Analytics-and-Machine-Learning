function prediction = KNNTesting(testImage,modelNN,K)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
LenNt=length(modelNN.neighbours)
for i=1:LenNt
    dis(i)=EuclideanDistance(testImage,modelNN.neighbours(i,:));
end
[Sorteddis,index]=sort(dis);
samp=modelNN.labels(index(1:K));
M=mode(samp);
prediction=M;
end