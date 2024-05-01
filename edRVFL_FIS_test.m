function [EVAL_Test,Testing_time,ProbScores] = edRVFL_FIS_test(testX,testY,model,option)

[Nsample,~] = size(testX);
activation = option.activation;

L = model.L;
w = model.w;
b= model.b;
beta = model.beta;
mu = model.mu;
sigma = model.sigma;
NF_Center=model.NF_Center;

A = cell(L,1); %for L hidden layers
ProbScores = cell(L,1); %depends on number of hidden layer
Alpha=option.Alpha;
NumFuzzyRule=option.NumFuzzyRule;
clus=option.clus;

[NF_Fuzzy,NF_DeFuzzy] = edRVFL_FIS_defuzzy_test(testX,NumFuzzyRule,Alpha,NF_Center);
A_input = testX;
A_input_NF=[testX,NF_Fuzzy];

tic
%% First Layer
for i = 1:L
    A1 = A_input_NF * w{i}+ repmat(b{i},Nsample,1);
    if option.renormal == 1
        if option.normal_type ==0
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i});
        end
    end
    if activation == 1
        A1 = sigmoid(A1,0,1);
    elseif activation == 2
        A1 = relu(A1);
    elseif activation == 3
        A1 = selu(A1);
    end
    if option.renormal == 1
        if option.normal_type ==1
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i});
        end
    end

    A1_temp1 = [testX,A1,ones(Nsample,1),NF_DeFuzzy];

    A{i} = A1_temp1;
    A_input_NF=[testX,A1,NF_Fuzzy];

    %% Calculate the testing accuracy
    beta_temp = beta{i};
    testY_temp = A1_temp1*beta_temp;

    %MajorityVoting
    [max_score,indx] = max(testY_temp,[],2);
    pred_idx(:,i) = indx;
end
EVAL_Test=majorityVoting(testY,pred_idx);

Testing_time = toc;
end