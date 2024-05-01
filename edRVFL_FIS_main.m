%% Reference %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Please cite the following paper if you are using this code.
% Reference: M. Sajid, M. Tanveer, and P. N. Suganthan. "Ensemble Deep Random Vector Functional Link Neural Network Based on Fuzzy Inference System”
% - Revision submitted in IEEE Transactions on Fuzzy Systems.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The experimental procedures are executed on a computing system possessing MATLAB R2023a software, Intel(R) Xeon(R) Platinum 8260 CPU @ 2.30GHz, 2301 Mhz, 24 Core(s),
% 48 Logical Processor(s) with 256 GB RAM on a Windows-10 operating platform.
%
% We have put a demo of the "edRVFL-FIS" model with the "cardiotocography_3clases" dataset
%
% For deatiled parameters setting, please refer "Ensemble Deep Random
% Vector Functional Link Neural Network Based on Fuzzy Inference System" paper.

%%
clc;
clear;
warning off all;
format compact;

%% Clustering Methods
% K-Means: clus=1
% Fuzzy C-Means: clus=2
% R-Means: clus=3
cluster=[1,2,3];
clus=1; % K-Means Cluster
option.clus=clus;

%% Data Preparation
split_ratio=0.8; nFolds=5; addpath(genpath('C:\Users\HP\OneDrive - IIT Indore\Desktop\NF-RVFL\Codes'))
temp_data1=load('cardiotocography_3clases.mat');

temp_data=temp_data1.cardiotocography_3clases;

[Cls,~,~] = unique(temp_data(:,end));
nclass = size(Cls,1);

trainX=temp_data(:,1:end-1); mean_X = mean(trainX,1); std_X = std(trainX);
trainX = bsxfun(@rdivide,trainX-repmat(mean_X,size(trainX,1),1),std_X);
All_Data=[trainX,temp_data(:,end)];

[samples,~]=size(All_Data);
rng('default')
test_start=floor(split_ratio*samples);
training_Data = All_Data(1:test_start-1,:); testing_Data = All_Data(test_start:end,:);
test_x=testing_Data(:,1:end-1); test_y=testing_Data(:,end);
train_x=training_Data(:,1:end-1); train_y=training_Data(:,end);

%% 0-1 hot encoding
U_dataY = 0:1:nclass-1;
% nclass=2;
testY_temp = zeros(numel(test_y),nclass);
for ii=1:nclass
    idx = test_y==U_dataY(ii);
    testY_temp(idx,ii)=1;
end
test_y=testY_temp;

U_dataY = 0:1:nclass-1;
trainY_temp = zeros(numel(train_y),nclass);
for ii=1:nclass
    idx = train_y==U_dataY(ii);
    trainY_temp(idx,ii)=1;
end
train_y=trainY_temp;

%% Demo Hyperparameter setting
option.activation = 1; %Sigmoid Activation function
option.scale = 1;
option.renormal=1;
option.normal_type=0; %(0 for batch normaliation and 1 for layer normalization)
option.L=7; %Number of hidden layers
option.N=810; %umber of hidden nodes
option.NumFuzzyRule=15; %Number of fuzzy layer nodes/centers/rules
option.C=1; %Regularization parameter

%% Randomly initializing parameters
option.Alpha=rand(size(train_x,2),option.NumFuzzyRule); % Randomly generating coefficients of THEN part of fuzzy rules for the fuzzy layer

%% Calling training function
[model,~,EVAL_Test,~,~] = MRVFL(train_x,train_y,test_x,test_y,option);

fprintf(1, 'Testing Accuracy of NF-RVFL model is: %f\n', EVAL_Test(1,1));

