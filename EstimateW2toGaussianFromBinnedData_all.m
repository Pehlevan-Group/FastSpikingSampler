function [w2] = EstimateW2toGaussianFromBinnedData_all(x, Sigma, mu, nBins)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

w2all = nan(size(x,2),1);

y = randn(size(x,1), size(x,2)) * chol(Sigma) + mu';

% Estimate the marginal distance for each dimension
for ind = 1:size(x,2)

    [~, edges] = histcounts(x(:,ind), nBins-2, "Normalization","pdf");
    edges = [-Inf, edges, Inf];

    xDisc = discretize(x(:,ind), edges);
    yDisc = discretize(y(:,ind), edges);

    w2all(ind) = ws_distance(xDisc, yDisc, 2);

end

w2 = (w2all);

end