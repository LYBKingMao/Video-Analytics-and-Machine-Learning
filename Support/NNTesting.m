function [prediction,distance] = NNTesting(testImage,modelNN)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
LenNt=length(modelNN.neighbours)
for i=1:LenNt
    loc(i)=EuclideanDistance(testImage,modelNN.neighbours(i,:));
end
[m,p]=min(loc);
distance=m;
prediction=modelNN.labels(p);