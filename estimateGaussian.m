function [mu sigma2] = estimateGaussian(X)
% This function estimates the parameters of a Gaussian distribution using the data in X
% The input X is the dataset with each n-dimensional data point in one row
% The output is an n-dimensional vector mu, the mean of the data set
% and the variances sigma^2, an n x 1 vector

[m, n] = size(X);
mu = zeros(n, 1);
sigma2 = zeros(n, 1);
mse=zeros(n,1);

% mu(i) contains the mean of the data for the i-th feature 
% sigma2(i) should contain variance of the i-th feature.

mu=mu+(1/m)*((sum(X))');
for i=1:n
  for j=1:m
    mse(i,1) = mse(i,1) + (X(j,i) - mu(i,1))^2;
endfor
  sigma2(i)=sigma2(i)+((1/m)*(mse(i)));
endfor

end
