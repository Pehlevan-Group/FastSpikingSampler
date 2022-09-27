clear;
close all;

%% Set parameters

% Number of parameters
p = 10;
n_p=p;
k=10;
n_n=k*n_p;

% Number of neurons
n = 100;

% Sampling timestep
dt = 1e-4;

% Compute actual time in seconds
tMax = 2;

T0=0.5;
T1=1.5;

% Fix number of timesteps
tMaxSteps = round(tMax/dt);

% Set decay parameter
tau_w = 0.020;
tau=tau_w;
eta = dt / tau;
tau_s=0.01*tau_w;

% Set number of repeats
nRep = 1000;



%Regularization params
alpha=sqrt(n_p*k);
lambda=sqrt(n_p*k);
%% Set up parallel pool

rng('shuffle');

poolObj = gcp('nocreate');
if isempty(poolObj)
    poolObj = parpool('local');
end


%% Set target distribution

% Mean of the target Gaussian distribution
% mu = zeros(p,1);

tOff = round(0.5/dt);
tOn = tMaxSteps - tOff;
maskVec = [zeros(tOff,1, 'logical'); ones(tOn,1, 'logical')];
mu = maskVec * ones(1,p)*6;

% Covariance matrix of target Gaussian distribution
sigma = 3;
rho = 0.75;
Sigma = sigma * (eye(p) + rho * (ones(p) - eye(p)));

% Define sensory inputs
A0=eye(n_p,n_p);
x0=pinv(Sigma*A0')*0*ones(n_p,1);

Mus_0 = 0*ones(n_p,1);
Mus_1 = 6*ones(n_p,1);                

A1=eye(n_p,n_p);
x1=pinv(Sigma*A1')*Mus_1;

%% Set weight matrix

% Define perfectly-balanced weight matrix
% A = randn(p,n/2);
% gammaNaive = [+A,-A];
% gammaGeom = sqrtm(Sigma) * [+A,-A];

Gamma=randn(p,n);

% Compute the initial membrane voltage
vInitNaive = (mu(1,:) / Sigma) * Gamma;
vInitGeom = (mu(1,:) / Sigma) * Gamma;

%%

isiBinEdges = (0:0.005:0.250)';

%% Run the simulation with naive geometry

tAll = tic;
fprintf('Naive sampler:\n');

totalSpikesNaive = nan(n, nRep);
isiCountNaive = zeros(length(isiBinEdges)-1, n, nRep);
isiMeanNaive = nan(n, nRep);
isiVarNaive = nan(n,nRep);

for indRep = 1:nRep

    % Start a timer
    tLoc = tic;


B=eye(n_p);
S=zeros(n_p);

start_voltage=zeros(n_n,1);
start_r=zeros(n_n,1);

% Run the sampler
[est_no_geo_0, volts_no_geo_0, rs_no_geo_0, os_no_geo_0] = EfficientBalancedSampling_classic(n_p,k,Sigma,A0,x0,Gamma,S,B,tau_w,tau_s,alpha,lambda,dt,T0,start_voltage,start_r);

end_voltage=volts_no_geo_0(:,end);
end_r=rs_no_geo_0(:,end);

[est_no_geo_1, volts_no_geo_1, rs_no_geo_1, os_no_geo_1] = EfficientBalancedSampling_classic(n_p,k,Sigma,A1,x1,Gamma,S,B,tau_w,tau_s,alpha,lambda,dt,T1,end_voltage,end_r);

full_sample_no_geo = cat(2, est_no_geo_0, est_no_geo_1);

full_spike_no_geo = cat(2, os_no_geo_0, os_no_geo_1);
full_spike_no_geo=full_spike_no_geo';

    % Compute per-neuron statistics
    totalSpikesNaive(:,indRep) = sum(full_spike_no_geo(maskVec,:),1);
    for indN = 1:n
        currISI = diff(find(full_spike_no_geo(maskVec,indN))) * dt;
        isiMeanNaive(indN,indRep) = mean(currISI);
        isiVarNaive(indN,indRep) = var(currISI);
        isiCountNaive(:,indN,indRep) = histcounts(currISI, isiBinEdges, 'normalization','count')';
    end

    fprintf('\trepeat %d of %d, %f seconds\n', indRep, nRep, toc(tLoc));
end

fprintf('\nFinished in %f seconds.\n', toc(tAll));

%% Run the simulation with natural geometry
tAll = tic;
fprintf('Geometry sampler:\n');

totalSpikesGeom = nan(n, nRep);
isiCountGeom = zeros(length(isiBinEdges)-1, n, nRep);
isiMeanGeom = nan(n, nRep);
isiVarGeom = nan(n,nRep);

for indRep = 1:nRep

    % Start a timer
    tLoc = tic;


B=sqrtm(Sigma);
S=zeros(n_p);

start_voltage=zeros(n_n,1);
start_r=zeros(n_n,1);

% Run the sampler
[est_geo_0, volts_geo_0, rs_geo_0, os_geo_0] = EfficientBalancedSampling_classic(n_p,k,Sigma,A0,x0,Gamma,S,B,tau_w,tau_s,alpha,lambda,dt,T0,start_voltage,start_r);

end_voltage=volts_geo_0(:,end);
end_r=rs_geo_0(:,end);

[est_geo_1, volts_geo_1, rs_geo_1, os_geo_1] = EfficientBalancedSampling_classic(n_p,k,Sigma,A1,x1,Gamma,S,B,tau_w,tau_s,alpha,lambda,dt,T1,end_voltage,end_r);

full_sample_geo = cat(2, est_geo_0, est_geo_1);

full_spike_geo = cat(2, os_geo_0, os_geo_1);
full_spike_geo=full_spike_geo';

    % Compute per-neuron statistics
    totalSpikesGeom(:,indRep) = sum(full_spike_geo(maskVec,:),1);
    for indN = 1:n
        currISI = diff(find(full_spike_geo(maskVec,indN))) * dt;
        isiMeanGeom(indN,indRep) = mean(currISI);
        isiVarGeom(indN,indRep) = var(currISI);
        isiCountGeom(:,indN,indRep) = histcounts(currISI, isiBinEdges, 'normalization','count')';
    end

    fprintf('\trepeat %d of %d, %f seconds\n', indRep, nRep, toc(tLoc));
end

fprintf('\nFinished in %f seconds.\n', toc(tAll));
% 
 % Set up plotting options

corder = [0.850980392156863, 0.372549019607843, 0.007843137254902; 0.458823529411765, 0.439215686274510, 0.701960784313725];


%% Plot rate histograms

% Firing rate histograms
binEdges = (0:10:200)';

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(corder);
hold on;
histogram(totalSpikesNaive / (tOn * dt), binEdges, 'normalization','probability', 'displaystyle','bar','edgecolor','none');
histogram(totalSpikesGeom / (tOn * dt), binEdges, 'normalization','probability', 'displaystyle','bar','edgecolor','none');
xlabel('spike rate (Hz)');
ylabel('relative frequency');
axis('square');
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

%% Plot ISI histograms

% ISI histogram across all neurons

isiBinCenters = isiBinEdges(1:end-1) + diff(isiBinEdges) / 2;

pNaive = sum(isiCountNaive, [2,3]);
pNaive = pNaive / sum(pNaive);
pGeom = sum(isiCountGeom, [2,3]);
pGeom = pGeom / sum(pGeom);

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(corder);
hold on;
bar(isiBinCenters, pNaive, 1, 'EdgeColor','None','FaceAlpha', 0.6);
bar(isiBinCenters, pGeom, 1, 'EdgeColor','None','FaceAlpha', 0.6);
xlabel('ISI (s)');
ylabel('relative frequency across trials, pooled across neurons');
axis('square');
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ylim([0 1]);
set(gca, 'yscale','log');
ConfAxis;

% ISI histogram for an example neuron
neuronInd = randi([1,n]);
pNaive = sum(isiCountNaive(:,neuronInd,:), 3);
pNaive = pNaive / sum(pNaive);
pGeom = sum(isiCountGeom(:,neuronInd,:), 3);
pGeom = pGeom / sum(pGeom);

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(corder);
hold on;
bar(isiBinCenters, pNaive, 1, 'EdgeColor','None','FaceAlpha', 0.6);
bar(isiBinCenters, pGeom, 1, 'EdgeColor','None','FaceAlpha', 0.6);
xlabel('ISI (s)');
ylabel(sprintf('relative frequency across trials, neuron %d', neuronInd));
axis('square');
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ylim([0 1]);
set(gca, 'yscale','log');
ConfAxis;

%% Plot histogram of CV of ISI across neurons

cvBinEdges = (0:0.1:2)';

% Combine across trials
isiMeanCombNaive = mean(isiMeanNaive, 2);
isiMeanCombGeom = mean(isiMeanGeom, 2);
isiVarCombNaive = mean(isiVarNaive, 2) + var(isiMeanNaive, 0, 2);
isiVarCombGeom = mean(isiVarGeom, 2) + var(isiMeanGeom, 0, 2);

isiCvNaive = sqrt(isiVarCombNaive) ./ isiMeanCombNaive;
isiCvGeom = sqrt(isiVarCombGeom) ./ isiMeanCombGeom;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(corder);
hold on;
histogram(isiCvNaive, cvBinEdges, 'normalization', 'probability','displaystyle','bar', 'edgecolor','none');
histogram(isiCvGeom, cvBinEdges, 'normalization', 'probability','displaystyle','bar', 'edgecolor','none');
xlabel('CV of ISI');
ylabel('relative frequency across neurons');
axis('square');
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ylim([0 1]);
set(gca, 'yscale','log');
ConfAxis;


%% Utility functions

function ConfAxis
    set(gca, 'FontSize', 16, 'LineWidth', 2, 'Box','off','TickDir','out');
end
