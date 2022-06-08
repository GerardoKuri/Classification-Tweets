clear all;
close all;
clc;
A=readmatrix('LabTweets.csv');
B=readmatrix('vectorizedTweets.csv');
B=B(2:end,2:end);
B=[A B];
train=B(1:800,:);
test=B(801:1000,:);

class=KNN(train,test,2);
confMat=confMatHD(test(:,1),transpose(class));
[Acc,Pre,Sen,F1Sc] = MatEval(confMat);