function [bestEpsilon bestF1] = selectThreshold(yval, pval)
% This funtion returns the best threshold (epsilon) to use for selecting outliers
% It finds the best threshold to use for selecting outliers based on the results 
% from a validation set (pval) and the ground truth (yval).

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
[m,n]=size(yval);

for epsilon = min(pval):stepsize:max(pval)

% tp is the number of true positives: the ground truth label says it’s an
% anomaly and our algorithm correctly classified it as an anomaly.
% fp is the number of false positives: the ground truth label says it’s not
% an anomaly, but our algorithm incorrectly classified it as an anomaly.
% fn is the number of false negatives: the ground truth label says it’s an
% anomaly, but our algorithm incorrectly classified it as not being anomalous

predictions = (pval < epsilon);
tp=sum((predictions == 1) & (yval == 1));
fn=sum((predictions == 0) & (yval == 1));
fp = sum((predictions == 1) & (yval == 0));
if ((tp+fp)!=0 && (tp+fn)!=0)
  prec= (tp)/(tp+fp);
  rec= (tp)/(tp+fn);
  F1=(2*prec*rec)/(prec+rec);
else
  F1=0;
end
  
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
    
   
end


end
