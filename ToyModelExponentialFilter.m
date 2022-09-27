function [sp, theta, r, v] = ToyModelExponentialFilter(n, p, S, mu, G, v0, tMax, eta, gain)

if nargin < 9
    gain = 1;
end

% Compute the weight matrix
Omega = G' * (S \ G);

% Allocate
v = nan(tMax, n);
sp = zeros(tMax,n);
r = zeros(tMax,n);

% Initialize membrane voltage (note that this choice must be made
% appropriately given the mean of the target distribution)
v(1,:) = v0;

% Determine whether mean is constant
constMean = (size(mu,1) == p && size(mu,2)==1) || (size(mu,2) == p && size(mu,1)==1);

% If not, compute projected current
if constMean
    muProj = eta * (mu / S) * G;
else
    muProj = ((mu(2:end,:) - (1-eta) * mu(1:end-1,:)) / S) * G;
end

% Iterate
for t = 2:tMax
    
    % Choose one neuron uniformly at random to propose a spike
    spikeProp = randi([1,n],1);

    % Compute acceptance ratio and perform Metropolis step
    a = min(1, exp(gain*(v(t-1,spikeProp) - Omega(spikeProp,spikeProp)/2)));
    if rand(1) <= a
        sp(t,spikeProp) = 1;
    end
    
    % Update membrane voltage
    if constMean
        v(t,:) = (1 - eta) * (v(t-1,:) - sp(t,:) * Omega) + muProj;
    else
        v(t,:) = (1 - eta) * (v(t-1,:) - sp(t,:) * Omega) + muProj(t-1,:);
    end

    % Update rate
    r(t,:) = (1 - eta) * r(t-1,:) + sp(t,:); 
end

% Compute estimate
theta = r * G';

end