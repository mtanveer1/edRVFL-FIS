function [F,G,center] = edRVFL_FIS_defuzzy_train(train_x,NumFuzzyRule,clus,Alpha)
%% Training starts
std = 1;
tic
Omega=zeros(size(train_x,1),NumFuzzyRule);

F = zeros(size(train_x,1), NumFuzzyRule); %Fuzzy Layer


%% Clustering Methods
cluster=clus;

if cluster==1
[~,center] = kmeans(train_x, NumFuzzyRule);
elseif cluster==2
[center,~] = fcm(train_x,NumFuzzyRule);
else
    Temptrain_x = randperm(length(train_x));
    indices = Temptrain_x(1:NumFuzzyRule);
    center = train_x(indices,:);
end
%%
for j = 1:size(train_x,1)   %calculating fuzzy membership "mew" - equation 10
    MF = exp(-(repmat(train_x(j,:), NumFuzzyRule,1) - center).^2/std);
    MF = prod(MF,2);  % equation 8
    MF = MF/sum(MF);  % equation 9
    F(j,:) = MF'.*(train_x(j,:)*Alpha);   % equation 11 & 12 and train_x(j,:)*b1 - equation 7
    Omega(j,:) = MF;
end

% F1 = [F,  0.1 * ones(size(F,1),1)];  %added bias column in fuzzy layer

%%%%%%%%%%%%%%%For the defuzzificaation%%%%%%%%%%%%%%%
G=Omega.*(train_x * Alpha);

end