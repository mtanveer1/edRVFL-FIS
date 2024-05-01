function [model,EVAL_Train,Training_time,ProbScores] = edRVFL_FIS_train(trainX,trainY,option)
[Nsample,Nfea] = size(trainX);
N = option.N;
L = option.L;
C = option.C;
activation = option.activation;
s = option.scale;  %scaling factor

A = cell(L,1); %for L hidden layers
beta = cell(L,1);
weights = cell(L,1);
biases = cell(L,1);
mu = cell(L,1);
sigma = cell(L,1);
ProbScores = cell(L,1); %depends on number of hidden layer
Alpha=option.Alpha;
NumFuzzyRule=option.NumFuzzyRule;
clus=option.clus;
%Fuzzy Layer Generation Starts
[NF_Fuzzy,NF_DeFuzzy,NF_Center] = edRVFL_FIS_defuzzy_train(trainX,NumFuzzyRule,clus,Alpha);%##
model.NF_Center=NF_Center;
[~,NF_Nfea] = size(NF_Fuzzy);
A_input_NF=[trainX,NF_Fuzzy];

tic
for i = 1:L

    if i==1
        w = s*2*rand(Nfea+NF_Nfea,N)-1;
    else
        w = s*2*rand(Nfea+NF_Nfea+N,N)-1;
    end

    b = s*rand(1,N);
    weights{i} = w;
    biases{i} = b;

    option.renormal=1;

    A1 = A_input_NF * w+repmat(b,Nsample,1);%%##
    if option.renormal == 1
        if option.normal_type ==0
            mu{i} = mean(A1,1);
            sigma{i} = std(A1);
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
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
            mu{i} = mean(A1,1);
            sigma{i} = std(A1);
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;layer normalization
        end
    end

    A1_temp1 = [trainX,A1,ones(Nsample,1),NF_DeFuzzy];
    beta1  = l2_weights(A1_temp1,trainY,C,Nsample);

    A{i} =  A1_temp1;
    beta{i} = beta1;

    A_input_NF=[trainX,A1,NF_Fuzzy];

    %% Calculate the training accuracy
    trainY_temp = A1_temp1*beta1;

    % MajorityVoting
    [~,indx] = max(trainY_temp,[],2);
    pred_idx(:,i) = indx;
end
EVAL_Train=majorityVoting(trainY,pred_idx);

Training_time = toc;

%%
model.L = L;
model.w = weights;
model.b = biases;
model.beta = beta;
model.mu = mu;
model.sigma = sigma;

end