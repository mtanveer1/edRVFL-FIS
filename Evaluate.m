function EVAL = Evaluate(ACTUAL,PREDICTED)
% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
idx = (ACTUAL()==1);
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = 100*(tp+tn)/N;
sensitivity = 100*tp_rate;
specificity = 100*tn_rate;
precision = 100*tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*sensitivity)/(precision + sensitivity));
gmean = 100*sqrt(tp_rate*tn_rate);
EVAL = [accuracy sensitivity specificity precision recall f_measure gmean tp tn fp fn];
