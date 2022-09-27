clear;
close all;

%% Set parameters

% Number of parameters
logPlist = (1:6);
pList = 2.^logPlist;

% Number of neurons per parameter (n = k*p)
kList = 10; 

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

%% Set up

rng('shuffle');

tSec = (0:tMaxSteps-1)' * dt;
fastMask = tSec > (tOn * dt) & tSec < (tOn * dt + tFast);

%% Set up parallel pool 

rng('shuffle');

poolObj = gcp('nocreate');
if isempty(poolObj)
    poolObj = parpool('local');
end

%% Sample using naive geometry

tAll = tic;
fprintf('Naive sampler:\n');

totalSpikesNaive = nan(length(pList),length(kList),nRep);
meanThetaMeanSteadyStateNaive = nan(length(pList),length(kList),nRep);
meanThetaVarSteadyStateNaive = nan(length(pList),length(kList),nRep);
thetaCovDistSteadyStateNaive = nan(length(pList),length(kList),nRep);
meanWassSteadyStateNaive = nan(length(pList),length(kList),nRep);

meanThetaMeanFastNaive = nan(length(pList),length(kList),nRep);
meanThetaVarFastNaive = nan(length(pList),length(kList),nRep);
thetaCovDistFastNaive = nan(length(pList),length(kList),nRep);
meanWassFastNaive = nan(length(pList),length(kList),nRep);

for indP = 1:length(pList)
    for indN = 1:length(kList)
        parfor indR = 1:nRep

            % Start a timer
            tInit = tic;

            % Mean of the target Gaussian distribution
            maskVec = [zeros(tOff,1, 'logical'); ones(tOn,1, 'logical')];
            mu = mu0 * maskVec * ones(1,pList(indP));

            % Covariance matrix of target Gaussian distribution
            Sigma = eye(pList(indP)) + rho * (ones(pList(indP)) - eye(pList(indP)));

            % Define perfectly-balanced weight matrix
            A = randn(pList(indP), pList(indP) * kList(indN)/2) / sqrt(pList(indP));
            G = [+A,-A];

            % Compute the initial membrane voltage
            v0 = (mu(1,:) / Sigma) * G;

            % Run the sampler
            [ spLoc, thetaLoc ] = ToyModelExponentialFilter(pList(indP) * kList(indN), pList(indP), Sigma, mu, G, v0, tMaxSteps, eta);

            % Compute stats
            totalSpikesNaive(indP, indN, indR) = mean(sum(spLoc(maskVec,:), 1),2);

            meanThetaMeanSteadyStateNaive(indP, indN, indR) = mean(mean(thetaLoc(maskVec,:), 1), 2);
            meanThetaVarSteadyStateNaive(indP, indN, indR) = mean(var(thetaLoc(maskVec,:), 0, 1), 2);
            meanThetaMeanFastNaive(indP, indN, indR) = mean(mean(thetaLoc(fastMask,:), 1), 2);
            meanThetaVarFastNaive(indP, indN, indR) = mean(var(thetaLoc(fastMask,:), 0, 1), 2);

            thetaMeanLoc = mean(thetaLoc(maskVec,:), 1)';
            thetaCovLoc = cov(thetaLoc(maskVec,:));
            thetaCovDistSteadyStateNaive(indP, indN, indR) = sum((thetaCovLoc - Sigma).^2, 'all');

            meanWassSteadyStateNaive(indP, indN, indR) = EstimateW2toGaussianFromBinnedData(thetaLoc(maskVec,:), Sigma, mu0 * ones(pList(indP),1), nBins);
            meanWassFastNaive(indP, indN, indR) = EstimateW2toGaussianFromBinnedData(thetaLoc(fastMask,:), Sigma, mu0 * ones(pList(indP),1), nBins);

            fprintf('\tp %d of %d, n %d of %d, repeat %d of %d, %f s\n', indP, length(pList), indN, length(kList), indR, nRep, toc(tInit));

        end
    end
end

fprintf('\nFinished in %f seconds.\n', toc(tAll));


%% Sample using optimized geometry

tAll = tic;
fprintf('With geometry:\n');


totalSpikesGeom = nan(length(pList),length(kList),nRep);
meanThetaMeanSteadyStateGeom = nan(length(pList),length(kList),nRep);
meanThetaVarSteadyStateGeom = nan(length(pList),length(kList),nRep);
thetaCovDistSteadyStateGeom = nan(length(pList),length(kList),nRep);
meanWassSteadyStateGeom = nan(length(pList),length(kList),nRep);

meanThetaMeanFastGeom = nan(length(pList),length(kList),nRep);
meanThetaVarFastGeom = nan(length(pList),length(kList),nRep);
thetaCovDistFastGeom = nan(length(pList),length(kList),nRep);
meanWassFastGeom = nan(length(pList),length(kList),nRep);

for indP = 1:length(pList)
    for indN = 1:length(kList)
        parfor indR = 1:nRep

            % Start a timer
            tInit = tic;

            % Mean of the target Gaussian distribution
            maskVec = [zeros(tOff,1, 'logical'); ones(tOn,1, 'logical')];
            mu = mu0 * maskVec * ones(1,pList(indP));

            % Covariance matrix of target Gaussian distribution
            Sigma = eye(pList(indP)) + rho * (ones(pList(indP)) - eye(pList(indP)));

            % Define perfectly-balanced weight matrix
            A = randn(pList(indP), pList(indP) * kList(indN)/2) / sqrt(pList(indP));
            G = sqrtm(Sigma) * [+A,-A];

            % Compute the initial membrane voltage
            v0 = (mu(1,:) / Sigma) * G;

            % Run the sampler
            [ spLoc, thetaLoc ] = ToyModelExponentialFilter(pList(indP) * kList(indN), pList(indP), Sigma, mu, G, v0, tMaxSteps, eta);

            % Compute stats
            totalSpikesGeom(indP, indN, indR) = mean(sum(spLoc(maskVec,:), 1),2);

            meanThetaMeanSteadyStateGeom(indP, indN, indR) = mean(mean(thetaLoc(maskVec,:), 1), 2);
            meanThetaVarSteadyStateGeom(indP, indN, indR) = mean(var(thetaLoc(maskVec,:), 0, 1), 2);
            meanThetaMeanFastGeom(indP, indN, indR) = mean(mean(thetaLoc(fastMask,:), 1), 2);
            meanThetaVarFastGeom(indP, indN, indR) = mean(var(thetaLoc(fastMask,:), 0, 1), 2);

            thetaMeanLoc = mean(thetaLoc(maskVec,:), 1)';
            thetaCovLoc = cov(thetaLoc(maskVec,:));
            thetaCovDistSteadyStateGeom(indP, indN, indR) = sum((thetaCovLoc - Sigma).^2, 'all');

            meanWassSteadyStateGeom(indP, indN, indR) = EstimateW2toGaussianFromBinnedData(thetaLoc(maskVec,:), Sigma, mu0 * ones(pList(indP),1), nBins);
            meanWassFastGeom(indP, indN, indR) = EstimateW2toGaussianFromBinnedData(thetaLoc(fastMask,:), Sigma, mu0 * ones(pList(indP),1), nBins);

            fprintf('\tp %d of %d, n %d of %d, repeat %d of %d, %f s\n', indP, length(pList), indN, length(kList), indR, nRep, toc(tInit));

        end
    end
end


fprintf('\nFinished in %f seconds.\n', toc(tAll));


%% Plot basic spike emission statistics

corder = [0.850980392156863, 0.372549019607843, 0.007843137254902; 0.458823529411765, 0.439215686274510, 0.701960784313725];

spikeRateNaive = squeeze(totalSpikesNaive(:,1,:) / (tOn * dt))';
spikeRateGeom = squeeze(totalSpikesGeom(:,1,:) / (tOn * dt))';
x = [squeeze(mean(spikeRateNaive,1))',squeeze(mean(spikeRateGeom,1))'];
ciNaive = bootci(nBoot, @mean, spikeRateNaive)';
ciGeom = bootci(nBoot, @mean, spikeRateGeom)';

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder);
PlotAsymmetricErrorPatch(logPlist', x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
xlabel('n_p');
xticks(logPlist);
% xticklabels(cellstr(num2str(logPlist', '2^{%d}')));
xticklabels(pList);
ylabel('mean spike rate (Hz)');
axis('square');
ConfAxis;


%% Plot steady-state statistics

% Mean
dimMeanThetaSteadyStateNaive = squeeze(meanThetaMeanSteadyStateNaive(:,1,:))';
dimMeanThetaSteadyStateGeom = squeeze(meanThetaMeanSteadyStateGeom(:,1,:))';
x = [squeeze(mean(dimMeanThetaSteadyStateNaive,1))',squeeze(mean(dimMeanThetaSteadyStateGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimMeanThetaSteadyStateNaive)';
ciGeom = bootci(nBoot, @mean, dimMeanThetaSteadyStateGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder);
PlotAsymmetricErrorPatch(logPlist', x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
plot(logPlist, mu0 * ones(length(pList),1), '--k', 'linewidth', 2);

xlabel('n_p');
xticks(logPlist);
% xticklabels(cellstr(num2str(logPlist', '2^{%d}')));
xticklabels(pList);
ylabel('mean steady-state mean across dimensions');
axis('square');
ConfAxis;

% Variance
dimVarThetaSteadyStateNaive = squeeze(meanThetaVarSteadyStateNaive(:,1,:))';
dimVarThetaSteadyStateGeom = squeeze(meanThetaVarSteadyStateGeom(:,1,:))';
x = [squeeze(mean(dimVarThetaSteadyStateNaive,1))',squeeze(mean(dimVarThetaSteadyStateGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimVarThetaSteadyStateNaive)';
ciGeom = bootci(nBoot, @mean, dimVarThetaSteadyStateGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder);
PlotAsymmetricErrorPatch(logPlist', x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
plot(logPlist, sigma * ones(length(pList),1), '--k', 'linewidth', 2);
xlabel('n_p');
xticks(logPlist);
% xticklabels(cellstr(num2str(logPlist', '2^{%d}')));
xticklabels(pList);
ylabel('mean steady-state variance across dimensions');
axis('square');
ConfAxis;

% 2-Wasserstein
dimMeanWassSteadyStateNaive = squeeze(meanWassSteadyStateNaive(:,1,:))';
dimMeanWassSteadyStateGeom = squeeze(meanWassSteadyStateGeom(:,1,:))';
x = [squeeze(mean(dimMeanWassSteadyStateNaive,1))',squeeze(mean(dimMeanWassSteadyStateGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimMeanWassSteadyStateNaive)';
ciGeom = bootci(nBoot, @mean, dimMeanWassSteadyStateGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder);
PlotAsymmetricErrorPatch(logPlist', x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
xlabel('n_p');
xticks(logPlist);
% xticklabels(cellstr(num2str(logPlist', '2^{%d}')));
xticklabels(pList);
ylabel('mean steady-state W_2 across dimensions');
axis('square');
ConfAxis;
set(gca, 'yscale','log');

%% Plot short-time statistics

% Mean
dimMeanThetaFastNaive = squeeze(meanThetaMeanFastNaive(:,1,:))';
dimMeanThetaFastGeom = squeeze(meanThetaMeanFastGeom(:,1,:))';
x = [squeeze(mean(dimMeanThetaFastNaive,1))',squeeze(mean(dimMeanThetaFastGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimMeanThetaFastNaive)';
ciGeom = bootci(nBoot, @mean, dimMeanThetaFastGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder);
PlotAsymmetricErrorPatch(logPlist', x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
plot(logPlist, mu0 * ones(length(pList),1), '--k', 'linewidth', 2);

xlabel('n_p');
xticks(logPlist);
% xticklabels(cellstr(num2str(logPlist', '2^{%d}')));
xticklabels(pList);
ylabel('mean short-time mean across dimensions');
axis('square');
ConfAxis;

% Variance
dimVarThetaFastNaive = squeeze(meanThetaVarFastNaive(:,1,:))';
dimVarThetaFastGeom = squeeze(meanThetaVarFastGeom(:,1,:))';
x = [squeeze(mean(dimVarThetaFastNaive,1))',squeeze(mean(dimVarThetaFastGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimVarThetaFastNaive)';
ciGeom = bootci(nBoot, @mean, dimVarThetaFastGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder);
PlotAsymmetricErrorPatch(logPlist', x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
plot(logPlist, sigma * ones(length(pList),1), '--k', 'linewidth', 2);
xlabel('n_p');
xticks(logPlist);
% xticklabels(cellstr(num2str(logPlist', '2^{%d}')));
xticklabels(pList);
ylabel('mean short-time variance across dimensions');
axis('square');
ConfAxis;

% 2-Wasserstein
dimMeanWassFastNaive = squeeze(meanWassFastNaive(:,1,:))';
dimMeanWassFastGeom = squeeze(meanWassFastGeom(:,1,:))';
x = [squeeze(mean(dimMeanWassFastNaive,1))',squeeze(mean(dimMeanWassFastGeom,1))'];
ciNaive = bootci(nBoot, @mean, dimMeanWassFastNaive)';
ciGeom = bootci(nBoot, @mean, dimMeanWassFastGeom)';
figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder);
PlotAsymmetricErrorPatch(logPlist', x, [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
xlabel('n_p');
xticks(logPlist);
% xticklabels(cellstr(num2str(logPlist', '2^{%d}')));
xticklabels(pList);
ylabel('mean short-time W_2 across dimensions');
axis('square');
ConfAxis;
set(gca, 'yscale','log');

%%

function ConfAxis
    set(gca, 'FontSize', 16, 'LineWidth', 2, 'Box','off','TickDir','out');
end

