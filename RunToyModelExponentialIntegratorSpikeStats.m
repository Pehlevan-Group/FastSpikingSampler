clear;
close all;

%% Set parameters

% Number of parameters
p = 10;

% Number of neurons
n = 100;

% Sampling timestep
dt = 1e-5;

% Compute actual time in seconds
tMax = 2;

% Fix number of timesteps
tMaxSteps = round(tMax/dt);

% Set decay parameter
tau = 0.020;
eta = dt / tau;

% Set number of repeats
nRep = 1000;

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
mu = maskVec * ones(1,p);

% Covariance matrix of target Gaussian distribution
sigma = 1;
rho = 0.75;
Sigma = sigma * (eye(p) + rho * (ones(p) - eye(p)));

%% Set weight matrix

% Define perfectly-balanced weight matrix
A = randn(p,n/2);
gammaNaive = [+A,-A];
gammaGeom = sqrtm(Sigma) * [+A,-A];

% Compute the initial membrane voltage
vInitNaive = (mu(1,:) / Sigma) * gammaNaive;
vInitGeom = (mu(1,:) / Sigma) * gammaGeom;

%%

isiBinEdges = (0:0.005:0.250)';

%% Run the simulation with naive geometry

tAll = tic;
fprintf('Naive sampler:\n');

totalSpikesNaive = nan(n, nRep);
isiCountNaive = zeros(length(isiBinEdges)-1, n, nRep);
isiMeanNaive = nan(n, nRep);
isiVarNaive = nan(n,nRep);

parfor indRep = 1:nRep

    % Start a timer
    tLoc = tic;

    % Run the sampler
    [spNaive, thetaNaive, rateNaive, vNaive] = ToyModelExponentialFilter(n, p, Sigma, mu, gammaNaive, vInitNaive, tMaxSteps, eta);

    % Compute per-neuron statistics
    totalSpikesNaive(:,indRep) = sum(spNaive(maskVec,:),1);
    for indN = 1:n
        currISI = diff(find(spNaive(maskVec,indN))) * dt;
        isiMeanNaive(indN,indRep) = mean(currISI);
        isiVarNaive(indN,indRep) = var(currISI);
        isiCountNaive(:,indN,indRep) = histcounts(currISI, isiBinEdges, 'normalization','count')';
    end

    fprintf('\trepeat %d of %d, %f seconds\n', indRep, nRep, toc(tLoc));
end

fprintf('\nFinished in %f seconds.\n', toc(tAll));

%% Run the simulation with natural geometry

tAll = tic;
fprintf('With geometry:\n');

totalSpikesGeom = nan(n, nRep);
isiCountGeom = zeros(length(isiBinEdges)-1, n, nRep);
isiMeanGeom = nan(n, nRep);
isiVarGeom = nan(n,nRep);

parfor indRep = 1:nRep

    % Start a timer
    tLoc = tic;


    [spGeom, thetaGeom, rateGeom, vGeom] = ToyModelExponentialFilter(n, p, Sigma, mu, gammaGeom, vInitGeom, tMaxSteps, eta);

    % Compute per-neuron statistics
    totalSpikesGeom(:,indRep) = sum(spGeom(maskVec,:),1);
    for indN = 1:n
        currISI = diff(find(spGeom(maskVec,indN))) * dt;
        isiMeanGeom(indN,indRep) = mean(currISI);
        isiVarGeom(indN,indRep) = var(currISI);
        isiCountGeom(:,indN,indRep) = histcounts(currISI, isiBinEdges, 'normalization','count')';
    end

    fprintf('\trepeat %d of %d, %f seconds\n', indRep, nRep, toc(tLoc));

end

fprintf('\nFinished in %f seconds.\n', toc(tAll));

%% Set up plotting options

corder = [0.850980392156863, 0.372549019607843, 0.007843137254902; 0.458823529411765, 0.439215686274510, 0.701960784313725];


%% Plot rate histograms

% Firing rate histograms
binEdges = (0:25:500)';

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
