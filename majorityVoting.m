function EVAL = majorityVoting(Y,pred_idx)

Nsample = size(Y,1);

[~, Ind_corrClass] = max(Y,[],2);

for i = 1:Nsample
    Y = pred_idx(i,:);
    a = unique(Y);
    [x,y] = hist(Y,a);
    [~,indx_temp] = max(x);
    indx(i,1) = y(indx_temp);
end

% acc = mean(indx == Ind_corrClass);

EVAL = Evaluate(indx,Ind_corrClass);

end