clear;
close all;

%% Set parameters

% Number of parameters
p = 10;
% p = 2;
% p = 32;

% Number of neurons
n = 100;
% n = 20;
% n = 320;

% Sampling timestep
dt = 1e-5;

% Compute actual time in seconds
tMax = 2;

% Fix number of timesteps
tMaxSteps = round(tMax/dt);

% Set decay parameter
tau = 0.020;
eta = dt / tau;

% Correlation
rho = 0.75;

% Marginal variance
sigma = 1;

% Stimulus time
mu0 = 1; 
tOff = round(0.5/dt);
tOn = tMaxSteps - tOff;

% Fix number of repeats
nRep = 100;

% Number of bins for estimation of 2-Wasserstein distance
nBins = 200;

% Set bin size for short-time statistics, in seconds
tFast = 0.05;

% Number of bootstraps (for error bar computation)
nBoot = 1000;

% Scale list
logScale = (-8:0.1:2)';
scaleList = 10.^logScale;

%% Set up parallel pool 

rng('shuffle');

poolObj = gcp('nocreate');
if isempty(poolObj)
    poolObj = parpool('local');
end

%% Run simulation without natural gradient

tAll = tic;
fprintf('Naive sampler:\n');

totalSpikesNaive = nan(n,length(scaleList),nRep);
meanThetaSteadyStateNaive = nan(p,length(scaleList),nRep);
varThetaSteadyStateNaive = nan(p,length(scaleList),nRep);
meanWassSteadyStateNaive = nan(length(scaleList),nRep);

meanThetaFastNaive = nan(p,length(scaleList),nRep);
varThetaFastNaive = nan(p,length(scaleList),nRep);
meanWassFastNaive = nan(length(scaleList),nRep);

%
for indScale = 1:length(scaleList)
    parfor indRep = 1:nRep

        % Start a timer
        tLoc = tic;

        % Mean of the target Gaussian distribution
        maskVec = [zeros(tOff,1, 'logical'); ones(tOn,1, 'logical')];
        mu = mu0 * maskVec * ones(1,p);
        
        % Covariance matrix of target Gaussian distribution
        Sigma = sigma * ((1-rho) * eye(p) + rho * ones(p));

        % Define perfectly-balanced weight matrix
        A = randn(p,n/2) * sqrt(scaleList(indScale));
        G = [+A,-A];

        % Compute the initial membrane voltage
        v0 = (mu(1,:) / Sigma) * G;

        % Run the sampler
        [spikeLoc, thetaLoc] = ToyModelExponentialFilter(n, p, Sigma, mu, G, v0, tMaxSteps, eta);
        
        % Compute statistics
        tSec = (0:tMaxSteps-1)' * dt;
        fastMask = tSec > (tOn * dt) & tSec < (tOn * dt + tFast);

        totalSpikesNaive(:, indScale, indRep) = sum(spikeLoc(maskVec,:), 1);

        meanThetaSteadyStateNaive(:, indScale, indRep) = mean(thetaLoc(maskVec,:), 1);
        varThetaSteadyStateNaive(:, indScale, indRep) = var(thetaLoc(maskVec,:), 0, 1);
        meanWassSteadyStateNaive(indScale, indRep) = EstimateW2toGaussianFromBinnedData(thetaLoc(maskVec,:), Sigma, mu0 * ones(p,1), nBins);

        meanThetaFastNaive(:, indScale, indRep) = mean(thetaLoc(fastMask,:), 1);
        varThetaFastNaive(:, indScale, indRep) = var(thetaLoc(fastMask,:), 0, 1);
        meanWassFastNaive(indScale, indRep) = EstimateW2toGaussianFromBinnedData(thetaLoc(fastMask,:), Sigma, mu0 * ones(p,1), nBins);

        fprintf('\tscale %d of %d, repeat %d of %d, %f seconds\n', indScale, length(scaleList), indRep, nRep, toc(tLoc));

    end
end

fprintf('\nFinished in %f seconds.\n', toc(tAll));

%% Run simulation with natural gradient

tAll = tic;
fprintf('With geometry:\n');

totalSpikesGeom = nan(n,length(scaleList),nRep);
meanThetaSteadyStateGeom = nan(p,length(scaleList),nRep);
varThetaSteadyStateGeom = nan(p,length(scaleList),nRep);
meanWassSteadyStateGeom = nan(length(scaleList),nRep);

meanThetaFastGeom = nan(p,length(scaleList),nRep);
varThetaFastGeom = nan(p,length(scaleList),nRep);
meanWassFastGeom = nan(length(scaleList),nRep);

%
for indScale = 1:length(scaleList)
    parfor indRep = 1:nRep

        % Start a timer
        tLoc = tic;

        % Mean of the target Gaussian distribution
        maskVec = [zeros(tOff,1, 'logical'); ones(tOn,1, 'logical')];
        mu = maskVec * ones(1,p);
        
        % Covariance matrix of target Gaussian distribution
        Sigma = sigma * ((1-rho) * eye(p) + rho * ones(p));

        % Define perfectly-balanced weight matrix
        A = randn(p,n/2) * sqrt(scaleList(indScale));
        G = sqrtm(Sigma) * [+A,-A];

        % Compute the initial membrane voltage
        v0 = (mu(1,:) / Sigma) * G;
        
        % Run the sampler
        [spikeLoc, thetaLoc] = ToyModelExponentialFilter(n, p, Sigma, mu, G, v0, tMaxSteps, eta);

        % Compute statistics
        tSec = (0:tMaxSteps-1)' * dt;
        fastMask = tSec > (tOn * dt) & tSec < (tOn * dt + tFast);

        totalSpikesGeom(:, indScale, indRep) = sum(spikeLoc(maskVec,:), 1);

        meanThetaSteadyStateGeom(:, indScale, indRep) = mean(thetaLoc(maskVec,:), 1);
        varThetaSteadyStateGeom(:, indScale, indRep) = var(thetaLoc(maskVec,:), 0, 1);
        meanWassSteadyStateGeom(indScale, indRep) = EstimateW2toGaussianFromBinnedData(thetaLoc(maskVec,:), Sigma, mu0 * ones(p,1), nBins);

        meanThetaFastGeom(:, indScale, indRep) = mean(thetaLoc(fastMask,:), 1);
        varThetaFastGeom(:, indScale, indRep) = var(thetaLoc(fastMask,:), 0, 1);
        meanWassFastGeom(indScale, indRep) = EstimateW2toGaussianFromBinnedData(thetaLoc(fastMask,:), Sigma, mu0 * ones(p,1), nBins);


        fprintf('\tscale %d of %d, repeat %d of %d, %f seconds\n', indScale, length(scaleList), indRep, nRep, toc(tLoc));

    end
end

fprintf('\nFinished in %f seconds.\n', toc(tAll));

%% Plot basic spiking statistics



corder = [0.850980392156863, 0.372549019607843, 0.007843137254902; 0.458823529411765, 0.439215686274510, 0.701960784313725];

% Plot total spikes
spikeRateNaive = squeeze(mean(totalSpikesNaive / (tOn * dt), 1))';
spikeRateGeom = squeeze(mean(totalSpikesGeom / (tOn * dt), 1))';
x = [squeeze(mean(spikeRateNaive,1))',squeeze(mean(spikeRateGeom,1))'];
ciNaive = bootci(nBoot, @mean, spikeRateNaive)';
ciGeom = bootci(nBoot, @mean, spikeRateGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
PlotAsymmetricErrorPatch(logScale, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
axis('square');
xticks((-8:8))
xticklabels(cellstr(num2str((-8:8)', '10^{%d}')));
xlabel('variance of random \Gamma')
ylabel('population-averaged spike rate (Hz)')
ConfAxis;
legend({'naive','natural'});


% Plot distribution over neurons of Fano factors of spike counts
binEdges = (0:25:500)';
binCenters = binEdges(1:end-1) + diff(binEdges) / 2;
pdfNaive = nan(length(binCenters), length(logScale));
pdfGeom = nan(length(binCenters), length(logScale));
for ind = 1:length(logScale)
    pdfNaive(:,ind) = histcounts(var(totalSpikesNaive(:,ind,:),0,3) ./ mean(totalSpikesNaive(:,ind,:),3),binEdges,'normalization','probability');
    pdfGeom(:,ind) = histcounts(var(totalSpikesGeom(:,ind,:),0,3) ./ mean(totalSpikesGeom(:,ind,:),3),binEdges,'normalization','probability');
end

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
cmp = interp1([1;0],[1,1,1; corder(1,:)], linspace(1,0,length(logScale))');
colororder(cmp);
colormap(cmp);
% caxis([0 1]);
cbar = colorbar;
ylabel(cbar, 'scale');
plot(binCenters, pdfNaive, 'linewidth', 2)
axis('square');
xlabel('Fano factor of spike count over stimulus presentation interval');
ylabel('relative frequency')
ConfAxis;
title('Naive geometry');

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
cmp = interp1([1;0],[1,1,1; corder(2,:)], linspace(1,0,length(logScale))');
colororder(cmp);
colormap(cmp);
% caxis([0 1]);
cbar = colorbar;
ylabel(cbar, 'scale');
plot(binCenters, pdfGeom, 'linewidth', 2)
axis('square');
xlabel('Fano factor of spike count over stimulus presentation interval');
ylabel('relative frequency')
ConfAxis;
title('Natural geometry');

% Plot distribution over neurons of Fano factors of spike rates
binEdges = (0:25:500)';
binCenters = binEdges(1:end-1) + diff(binEdges) / 2;
pdfNaive = nan(length(binCenters), length(logScale));
pdfGeom = nan(length(binCenters), length(logScale));
for ind = 1:length(logScale)
    pdfNaive(:,ind) = histcounts(var(totalSpikesNaive(:,ind,:) / (tOn * dt),0,3) ./ mean(totalSpikesNaive(:,ind,:)/ (tOn * dt),3),binEdges,'normalization','probability');
    pdfGeom(:,ind) = histcounts(var(totalSpikesGeom(:,ind,:)/ (tOn * dt),0,3) ./ mean(totalSpikesGeom(:,ind,:)/ (tOn * dt),3),binEdges,'normalization','probability');
end

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
cmp = interp1([1;0],[1,1,1; corder(1,:)], linspace(1,0,length(logScale))');
colororder(cmp);
colormap(cmp);
caxis([0 1]);
cbar = colorbar;
ylabel(cbar, 'scale');
plot(binCenters, pdfNaive, 'linewidth', 2)
axis('square');
xlabel('Fano factor of spike rate over stimulus presentation interval (Hz)');
ylabel('relative frequency')
ConfAxis;
title('Naive geometry');

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
cmp = interp1([1;0],[1,1,1; corder(2,:)], linspace(1,0,length(logScale))');
colororder(cmp);
colormap(cmp);
caxis([0 1]);
cbar = colorbar;
ylabel(cbar, 'scale');
plot(binCenters, pdfGeom, 'linewidth', 2)
axis('square');
xlabel('Fano factor of spike rate over stimulus presentation interval (Hz)');
ylabel('relative frequency')
ConfAxis;
title('Natural geometry');


%% Plot steady-state statistics

% Plot dimension-averaged mean
dimMeanThetaSteadyStateNaive = squeeze(mean(meanThetaSteadyStateNaive,1))';
dimMeanThetaSteadyStateGeom = squeeze(mean(meanThetaSteadyStateGeom,1))';
x = [squeeze(mean(dimMeanThetaSteadyStateNaive,1))',squeeze(mean(dimMeanThetaSteadyStateGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimMeanThetaSteadyStateNaive)';
ciGeom = bootci(nBoot, @mean, dimMeanThetaSteadyStateGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
PlotAsymmetricErrorPatch(logScale, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on;
plot(logScale, mu0 * ones(length(logScale),1), '--k', 'linewidth', 2);
axis('square');
xticks((-8:8))
xticklabels(cellstr(num2str((-8:8)', '10^{%d}')));
xlabel('variance of random \Gamma')
ylabel('steady-state dimension-averaged readout mean')
ConfAxis;
legend({'naive','natural'});

% Plot dimension-averaged variance
dimVarThetaSteadyStateNaive = squeeze(mean(varThetaSteadyStateNaive,1))';
dimVarThetaSteadyStateGeom = squeeze(mean(varThetaSteadyStateGeom,1))';
x = [squeeze(mean(dimVarThetaSteadyStateNaive,1))',squeeze(mean(dimVarThetaSteadyStateGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimVarThetaSteadyStateNaive)';
ciGeom = bootci(nBoot, @mean, dimVarThetaSteadyStateGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
PlotAsymmetricErrorPatch(logScale, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on;
plot(logScale, sigma * ones(length(logScale),1), '--k', 'linewidth', 2);
axis('square');
xticks((-8:8))
xticklabels(cellstr(num2str((-8:8)', '10^{%d}')));
xlabel('variance of random \Gamma')
ylabel('steady-state dimension-averaged readout variance')
ConfAxis;
legend({'naive','natural'});

% Plot mean W_2
dimVarThetaSteadyStateNaive = squeeze(mean(varThetaSteadyStateNaive,1))';
dimVarThetaSteadyStateGeom = squeeze(mean(varThetaSteadyStateGeom,1))';
x = [squeeze(mean(meanWassSteadyStateNaive,2)),squeeze(mean(meanWassSteadyStateGeom,2))];
ciNaive = bootci(nBoot, @mean, meanWassSteadyStateNaive')';
ciGeom = bootci(nBoot, @mean, meanWassSteadyStateGeom')';
figure('Position',[200,500,500,700],'WindowStyle','docked');
PlotAsymmetricErrorPatch(logScale, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
axis('square');
xticks((-8:8))
xticklabels(cellstr(num2str((-8:8)', '10^{%d}')));
xlabel('variance of random \Gamma')
ylabel('steady-state dimension-averaged W_2 distance');
set(gca, 'yscale', 'log');
ConfAxis;
legend({'naive','natural'});

%% Plot short-time statistics

% Plot dimension-averaged mean
dimMeanThetaFastNaive = squeeze(mean(meanThetaFastNaive,1))';
dimMeanThetaFastGeom = squeeze(mean(meanThetaFastGeom,1))';
x = [squeeze(mean(dimMeanThetaFastNaive,1))',squeeze(mean(dimMeanThetaFastGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimMeanThetaFastNaive)';
ciGeom = bootci(nBoot, @mean, dimMeanThetaFastGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
PlotAsymmetricErrorPatch(logScale, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on;
plot(logScale, mu0 * ones(length(logScale),1), '--k', 'linewidth', 2);
axis('square');
xticks((-8:8))
xticklabels(cellstr(num2str((-8:8)', '10^{%d}')));
xlabel('variance of random \Gamma')
ylabel('short-time dimension-averaged readout mean')
ConfAxis;
legend({'naive','natural'});

% Plot dimension-averaged variance
dimVarThetaFastNaive = squeeze(mean(varThetaFastNaive,1))';
dimVarThetaFastGeom = squeeze(mean(varThetaFastGeom,1))';
x = [squeeze(mean(dimVarThetaFastNaive,1))',squeeze(mean(dimVarThetaFastGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimVarThetaFastNaive)';
ciGeom = bootci(nBoot, @mean, dimVarThetaFastGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
PlotAsymmetricErrorPatch(logScale, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on;
plot(logScale, sigma * ones(length(logScale),1), '--k', 'linewidth', 2);
axis('square');
xticks((-8:8))
xticklabels(cellstr(num2str((-8:8)', '10^{%d}')));
xlabel('variance of random \Gamma')
ylabel('short-time dimension-averaged readout variance')
ConfAxis;
legend({'naive','natural'});

% Plot mean W_2
dimVarThetaFastNaive = squeeze(mean(varThetaFastNaive,1))';
dimVarThetaFastGeom = squeeze(mean(varThetaFastGeom,1))';
x = [squeeze(mean(meanWassFastNaive,2)),squeeze(mean(meanWassFastGeom,2))];
ciNaive = bootci(nBoot, @mean, meanWassFastNaive')';
ciGeom = bootci(nBoot, @mean, meanWassFastGeom')';
figure('Position',[200,500,500,700],'WindowStyle','docked');
PlotAsymmetricErrorPatch(logScale, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
axis('square');
xticks((-8:8))
xticklabels(cellstr(num2str((-8:8)', '10^{%d}')));
xlabel('variance of random \Gamma')
ylabel('short-time dimension-averaged W_2 distance');
set(gca, 'yscale', 'log');
ConfAxis;
legend({'naive','natural'});

%% 

function ConfAxis
    set(gca, 'FontSize', 16, 'LineWidth', 2, 'Box','off','TickDir','out');
end
