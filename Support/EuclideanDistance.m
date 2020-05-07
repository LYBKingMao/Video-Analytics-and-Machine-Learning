function dEuc = EuclideanDistance(sample1,sample2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
len=length(sample1);
for i=1:len
sum1(i)=power((sample1(i)-sample2(i)),2);
end
summ=sum(sum1);
dEuc=sqrt(summ);