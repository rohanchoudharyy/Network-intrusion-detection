clear ; close all; clc
fprintf('... Reading the dataset ....');

% Preparation of Training Set, Validation Set, and Test Set
% The Training Set should have 60% (approx.) of the total number of instances present in the dataset.
% The Validation Set should have 20% (approx.) of the total number of instances containing both 
% anomalous and non-anomalous examples (as per the labels).
% The Test Set, containing the remaining instances should also have both anomalous and non-anomalous
% examples (as per labels).

raw=csvread('rawdata4.csv');
X=raw(1:600,11:13);
y=raw(1:600,15);
Xval=raw(601:900,11:13);
yval=raw(601:900,15);
Xtest=raw(901:1218,11:13);
ytest=raw(901:1218,15);

%  For this data we have assumed a Gaussian distribution.
%  We first estimate the parameters of our assumed Gaussian distribution, 
%  then compute the probabilities for each of the points and then visualize 
%  both the overall distribution and where each of the points falls in 
%  terms of that distribution.

%  Estimate mean and covariance
[mu sigma2] = estimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2);

%  Now you will find a good epsilon threshold using a cross-validation set
%  probabilities given the estimated Gaussian distribution


pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon F1] = selectThreshold(yval, pval);

fprintf('\n\nBest epsilon found :%e\n', epsilon);
fprintf('Best F1 score on CV Set:  %f\n', F1);
fprintf('Anomalies found: %d\n\n', sum(pval < epsilon));

% tp is the number of true positives: the ground truth label says it’s an
% anomaly and our algorithm correctly classified it as an anomaly.
% fp is the number of false positives: the ground truth label says it’s not
% an anomaly, but our algorithm incorrectly classified it as an anomaly.
% fn is the number of false negatives: the ground truth label says it’s an
% anomaly, but our algorithm incorrectly classified it as not being anomalous

fprintf('... Accuracy parameters on cross validation dataset ...\n\n');
predictions = (pval < epsilon);
tp=sum((predictions == 1) & (yval == 1));
fn=sum((predictions == 0) & (yval == 1));
fp = sum((predictions == 1) & (yval == 0));
true_positive=tp
false_negative=fn
false_positive=fp
prec= (tp)/(tp+fp);
rec= (tp)/(tp+fn)
F1_score=(2*prec*rec)/(prec+rec)

fprintf('\n... Accuracy parameters on test dataset ...\n\n');
ptest = multivariateGaussian(Xtest, mu, sigma2);
predictions = (ptest < epsilon);
tp=sum((predictions == 1) & (ytest == 1));
fn=sum((predictions == 0) & (ytest == 1));
fp = sum((predictions == 1) & (ytest == 0));
true_positive=tp
false_negative=fn
false_positive=fp
prec= (tp)/(tp+fp);
rec= (tp)/(tp+fn)
F1_score=(2*prec*rec)/(prec+rec)

% test on new data


