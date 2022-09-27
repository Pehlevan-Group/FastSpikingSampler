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

% Correlation
rhoList = (0:0.01:0.99)';

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

%% Set up parallel pool 

rng('shuffle');

poolObj = gcp('nocreate');
if isempty(poolObj)
    poolObj = parpool('local');
end

%% Run simulation without natural gradient

tAll = tic;
fprintf('Naive sampler:\n');

totalSpikesNaive = nan(n,length(rhoList),nRep);
meanThetaSteadyStateNaive = nan(p,length(rhoList),nRep);
varThetaSteadyStateNaive = nan(p,length(rhoList),nRep);
meanWassSteadyStateNaive = nan(length(rhoList),nRep);

meanThetaFastNaive = nan(p,length(rhoList),nRep);
varThetaFastNaive = nan(p,length(rhoList),nRep);
meanWassFastNaive = nan(length(rhoList),nRep);

%
for indRho = 1:length(rhoList)
    parfor indRep = 1:nRep

        % Start a timer
        tLoc = tic;

        % Mean of the target Gaussian distribution
        maskVec = [zeros(tOff,1, 'logical'); ones(tOn,1, 'logical')];
        mu = mu0 * maskVec * ones(1,p);
        
        % Covariance matrix of target Gaussian distribution
        Sigma = sigma * ((1 - rhoList(indRho)) * eye(p) + rhoList(indRho) * ones(p));

        % Define perfectly-balanced weight matrix
%         A = randn(p,n/2) / sqrt(n);
        A = randn(p,n/2);
        G = [+A,-A];

        % Compute the initial membrane voltage
        v0 = (mu(1,:) / Sigma) * G;

        % Run the sampler
        [spikeLoc, thetaLoc] = ToyModelExponentialFilter(n, p, Sigma, mu, G, v0, tMaxSteps, eta);

        % Compute statistics
        tSec = (0:tMaxSteps-1)' * dt;
        fastMask = tSec > (tOn * dt) & tSec < (tOn * dt + tFast);

        totalSpikesNaive(:, indRho, indRep) = sum(spikeLoc(maskVec,:), 1);

        meanThetaSteadyStateNaive(:, indRho, indRep) = mean(thetaLoc(maskVec,:), 1);
        varThetaSteadyStateNaive(:, indRho, indRep) = var(thetaLoc(maskVec,:), 0, 1);
        meanWassSteadyStateNaive(indRho, indRep) = EstimateW2toGaussianFromBinnedData(thetaLoc(maskVec,:), Sigma, mu0 * ones(p,1), nBins);

        meanThetaFastNaive(:, indRho, indRep) = mean(thetaLoc(fastMask,:), 1);
        varThetaFastNaive(:, indRho, indRep) = var(thetaLoc(fastMask,:), 0, 1);
        meanWassFastNaive(indRho, indRep) = EstimateW2toGaussianFromBinnedData(thetaLoc(fastMask,:), Sigma, mu0 * ones(p,1), nBins);

        fprintf('\trho %d of %d, repeat %d of %d, %f seconds\n', indRho, length(rhoList), indRep, nRep, toc(tLoc));

    end
end

fprintf('\nFinished in %f seconds.\n', toc(tAll));

%% Run simulation with natural gradient

tAll = tic;
fprintf('With geometry:\n');

totalSpikesGeom = nan(n,length(rhoList),nRep);
meanThetaSteadyStateGeom = nan(p,length(rhoList),nRep);
varThetaSteadyStateGeom = nan(p,length(rhoList),nRep);
meanWassSteadyStateGeom = nan(length(rhoList),nRep);

meanThetaFastGeom = nan(p,length(rhoList),nRep);
varThetaFastGeom = nan(p,length(rhoList),nRep);
meanWassFastGeom = nan(length(rhoList),nRep);

%
for indRho = 1:length(rhoList)
    parfor indRep = 1:nRep

        % Start a timer
        tLoc = tic;

        % Mean of the target Gaussian distribution
        maskVec = [zeros(tOff,1, 'logical'); ones(tOn,1, 'logical')];
        mu = maskVec * ones(1,p);
        
        % Covariance matrix of target Gaussian distribution
        Sigma = sigma * ((1 - rhoList(indRho)) * eye(p) + rhoList(indRho) * ones(p));

        % Define perfectly-balanced weight matrix
%         A = randn(p,n/2) / sqrt(n);
        A = randn(p,n/2);
        G = sqrtm(Sigma) * [+A,-A];

        % Compute the initial membrane voltage
        v0 = (mu(1,:) / Sigma) * G;
        
        % Run the sampler
        [spikeLoc, thetaLoc] = ToyModelExponentialFilter(n, p, Sigma, mu, G, v0, tMaxSteps, eta);

        % Compute statistics
        tSec = (0:tMaxSteps-1)' * dt;
        fastMask = tSec > (tOn * dt) & tSec < (tOn * dt + tFast);

        totalSpikesGeom(:, indRho, indRep) = sum(spikeLoc(maskVec,:), 1);

        meanThetaSteadyStateGeom(:, indRho, indRep) = mean(thetaLoc(maskVec,:), 1);
        varThetaSteadyStateGeom(:, indRho, indRep) = var(thetaLoc(maskVec,:), 0, 1);
        meanWassSteadyStateGeom(indRho, indRep) = EstimateW2toGaussianFromBinnedData(thetaLoc(maskVec,:), Sigma, mu0 * ones(p,1), nBins);

        meanThetaFastGeom(:, indRho, indRep) = mean(thetaLoc(fastMask,:), 1);
        varThetaFastGeom(:, indRho, indRep) = var(thetaLoc(fastMask,:), 0, 1);
        meanWassFastGeom(indRho, indRep) = EstimateW2toGaussianFromBinnedData(thetaLoc(fastMask,:), Sigma, mu0 * ones(p,1), nBins);


        fprintf('\trho %d of %d, repeat %d of %d, %f seconds\n', indRho, length(rhoList), indRep, nRep, toc(tLoc));

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
PlotAsymmetricErrorPatch(rhoList, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
axis('square');
xlabel('\rho')
ylabel('population-averaged spike rate (Hz)')
ConfAxis;
legend({'naive','natural'});


% Plot distribution over neurons of Fano factors of spike counts
binEdges = (0:25:500)';
binCenters = binEdges(1:end-1) + diff(binEdges) / 2;
pdfNaive = nan(length(binCenters), length(rhoList));
pdfGeom = nan(length(binCenters), length(rhoList));
for ind = 1:length(rhoList)
    pdfNaive(:,ind) = histcounts(var(totalSpikesNaive(:,ind,:),0,3) ./ mean(totalSpikesNaive(:,ind,:),3),binEdges,'normalization','probability');
    pdfGeom(:,ind) = histcounts(var(totalSpikesGeom(:,ind,:),0,3) ./ mean(totalSpikesGeom(:,ind,:),3),binEdges,'normalization','probability');
end

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
cmp = interp1([1;0],[1,1,1; corder(1,:)], linspace(1,0,length(rhoList))');
colororder(cmp);
colormap(cmp);
caxis([0 1]);
cbar = colorbar;
ylabel(cbar, '\rho');
plot(binCenters, pdfNaive, 'linewidth', 2)
axis('square');
xlabel('Fano factor of spike count over stimulus presentation interval');
ylabel('relative frequency')
ConfAxis;
title('Naive geometry');

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
cmp = interp1([1;0],[1,1,1; corder(2,:)], linspace(1,0,length(rhoList))');
colororder(cmp);
colormap(cmp);
caxis([0 1]);
cbar = colorbar;
ylabel(cbar, '\rho');
plot(binCenters, pdfGeom, 'linewidth', 2)
axis('square');
xlabel('Fano factor of spike count over stimulus presentation interval');
ylabel('relative frequency')
ConfAxis;
title('Natural geometry');

% Plot distribution over neurons of Fano factors of spike rates
binEdges = (0:25:500)';
binCenters = binEdges(1:end-1) + diff(binEdges) / 2;
pdfNaive = nan(length(binCenters), length(rhoList));
pdfGeom = nan(length(binCenters), length(rhoList));
for ind = 1:length(rhoList)
    pdfNaive(:,ind) = histcounts(var(totalSpikesNaive(:,ind,:) / (tOn * dt),0,3) ./ mean(totalSpikesNaive(:,ind,:)/ (tOn * dt),3),binEdges,'normalization','probability');
    pdfGeom(:,ind) = histcounts(var(totalSpikesGeom(:,ind,:)/ (tOn * dt),0,3) ./ mean(totalSpikesGeom(:,ind,:)/ (tOn * dt),3),binEdges,'normalization','probability');
end

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
cmp = interp1([1;0],[1,1,1; corder(1,:)], linspace(1,0,length(rhoList))');
colororder(cmp);
colormap(cmp);
caxis([0 1]);
cbar = colorbar;
ylabel(cbar, '\rho');
plot(binCenters, pdfNaive, 'linewidth', 2)
axis('square');
xlabel('Fano factor of spike rate over stimulus presentation interval (Hz)');
ylabel('relative frequency')
ConfAxis;
title('Naive geometry');

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
cmp = interp1([1;0],[1,1,1; corder(2,:)], linspace(1,0,length(rhoList))');
colororder(cmp);
colormap(cmp);
caxis([0 1]);
cbar = colorbar;
ylabel(cbar, '\rho');
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
v
figure('Position',[200,500,500,700],'WindowStyle','docked');
PlotAsymmetricErrorPatch(rhoList, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on;
plot(rhoList, mu0 * ones(length(rhoList),1), '--k', 'linewidth', 2);
axis('square');
xlabel('\rho')
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
PlotAsymmetricErrorPatch(rhoList, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on;
plot(rhoList, sigma * ones(length(rhoList),1), '--k', 'linewidth', 2);
axis('square');
xlabel('\rho')
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
PlotAsymmetricErrorPatch(rhoList, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
axis('square');
xlabel('\rho');
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
PlotAsymmetricErrorPatch(rhoList, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on;
plot(rhoList, mu0 * ones(length(rhoList),1), '--k', 'linewidth', 2);
axis('square');
xlabel('\rho')
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
PlotAsymmetricErrorPatch(rhoList, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on;
plot(rhoList, sigma * ones(length(rhoList),1), '--k', 'linewidth', 2);
axis('square');
xlabel('\rho')
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
PlotAsymmetricErrorPatch(rhoList, x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
axis('square');
xlabel('\rho');
ylabel('short-time dimension-averaged W_2 distance');
set(gca, 'yscale', 'log');
ConfAxis;
legend({'naive','natural'});

%% 

function ConfAxis
    set(gca, 'FontSize', 16, 'LineWidth', 2, 'Box','off','TickDir','out');
end
