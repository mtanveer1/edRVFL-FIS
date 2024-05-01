function [Model,EVAL_Train,EVAL_Test,TrainingTime,TestingTime] = ...
    MRVFL(trainX,trainY,testX,testY,option)

seed = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(seed);

% Train RVFL
% [Model,train_acc,TrainingTime,~] = MRVFLtrain(trainX,trainY,option); %For without Evaluate matrices
[Model,EVAL_Train,TrainingTime,~] = edRVFL_FIS_train(trainX,trainY,option);

% Using trained model, predict the testing data
% [TestAcc,TestingTime,~] = MRVFLpredict(testX,testY,Model,option);
[EVAL_Test,TestingTime,~] = edRVFL_FIS_test(testX,testY,Model,option);

end
%EOF