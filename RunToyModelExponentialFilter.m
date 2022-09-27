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

%% Compute the initial membrane voltage


vInitNaive = (mu(1,:) / Sigma) * gammaNaive;
vInitGeom = (mu(1,:) / Sigma) * gammaGeom;


%% Run the sampler

tic;
[spNaive, thetaNaive, rateNaive, vNaive] = ToyModelExponentialFilter(n, p, Sigma, mu, gammaNaive, vInitNaive, tMaxSteps, eta);
toc;

tic;
[spGeom, thetaGeom, rateGeom, vGeom] = ToyModelExponentialFilter(n, p, Sigma, mu, gammaGeom, vInitGeom, tMaxSteps, eta);
toc;

%% Plot basic quantifications of the results

tSec = (0:tMaxSteps-1)' * dt;

load('blueRedColorMap.mat','cmpBlueRed');

corder = [0.850980392156863, 0.372549019607843, 0.007843137254902; 0.458823529411765, 0.439215686274510, 0.701960784313725];

interpNaive = interp1([1;0],[1,1,1; corder(1,:)], linspace(1,0,p)');
interpGeom = interp1([1;0],[1,1,1; corder(2,:)], linspace(1,0,p)');

%% Plot moving averages of parameter estimates

tauAverage = 0.01;
downsampleFactor = 100;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(interpNaive);
plot(downsample(tSec,downsampleFactor), downsample(fast_moving_average(thetaNaive,tauAverage/dt), downsampleFactor),'linewidth',1);
hold on;
plot(downsample(tSec, downsampleFactor), downsample(mu, downsampleFactor), '--k', 'linewidth', 1);
legend(cellstr(num2str((1:p)', '\\theta_{%d}')))
axis('square');
xlabel('time (s)')
ylabel('moving average of parameter estimate (arb units)');
title(sprintf('Naive geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(interpGeom);
plot(downsample(tSec,downsampleFactor), downsample(fast_moving_average(thetaGeom,tauAverage/dt), downsampleFactor),'linewidth',1);
hold on;
plot(downsample(tSec, downsampleFactor), downsample(mu, downsampleFactor), '--k', 'linewidth', 1);
legend(cellstr(num2str((1:p)', '\\theta_{%d}')))
axis('square');
xlabel('time (s)')
ylabel('moving average of parameter estimate (arb units)');
title(sprintf('Natural geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder([interpNaive; interpGeom]);
hold on;
plot(downsample(tSec,downsampleFactor), downsample(fast_moving_average(thetaNaive,tauAverage/dt), downsampleFactor),'linewidth',1);
plot(downsample(tSec,downsampleFactor), downsample(fast_moving_average(thetaGeom,tauAverage/dt), downsampleFactor),'linewidth',1);
plot(downsample(tSec, downsampleFactor), downsample(mu, downsampleFactor), '--k', 'linewidth', 1);
axis('square');
xlabel('time (s)')
ylabel('moving average of parameter estimate (arb units)');
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

%% Plot marginal distributions of parameter estimates

binEdges = (-10:0.25:10);

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(interpNaive);
hold on;
for ind = 1:p
    histogram(thetaNaive(maskVec,ind),binEdges,'displaystyle','bar','normalization','probability','edgecolor','none');
end
axis('square');
xlabel('\theta_i');
ylabel('marginal relative frequency');
legend(cellstr(num2str((1:p)', '\\theta_{%d}')));
title(sprintf('Naive geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(interpGeom);
hold on;
for ind = 1:p
    histogram(thetaGeom(maskVec,ind),binEdges,'displaystyle','bar','normalization','probability','edgecolor','none');
end
axis('square');
xlabel('\theta_i');
ylabel('marginal relative frequency');
legend(cellstr(num2str((1:p)', '\\theta_{%d}')));
title(sprintf('Natural geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder([interpNaive; interpGeom]);
hold on;
for ind = 1:p
    histogram(thetaNaive(maskVec,ind),binEdges,'displaystyle','bar','normalization','probability','edgecolor','none');
end
for ind = 1:p
    histogram(thetaGeom(maskVec,ind),binEdges,'displaystyle','bar','normalization','probability','edgecolor','none');
end
axis('square');
xlabel('\theta_i');
ylabel('marginal relative frequency');
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

%% Plot spike rate distributions

% Compute overall spike rate when stimulus is present
spRateNaive = sum(spNaive(maskVec,:),1) / (tOn * dt);
spRateGeom = sum(spGeom(maskVec,:),1)  / (tOn * dt);

binEdges = (0:25:500)';

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(corder);
hold on;
histogram(spRateNaive, binEdges, 'normalization','probability', 'displaystyle','bar','edgecolor','none');
histogram(spRateGeom, binEdges, 'normalization','probability', 'displaystyle','bar','edgecolor','none');
xlabel('spike rate (Hz)');
ylabel('relative frequency');
axis('square');
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

%% Plot spike rasters

tRange = [1,1.1];

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder(1,:));
for ind = 1:n
    currSpikes = tSec(spNaive(:,ind)==1);
    plot(currSpikes, ind * ones(length(currSpikes),1), '|', 'MarkerSize', 10, 'LineWidth',1.5);
end
axis('square');
xlabel('time (s)');
ylabel('neurons');
yticks([]);
xlim(tRange);
title(sprintf('Naive geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
colororder(corder(2,:));
for ind = 1:n
    currSpikes = tSec(spGeom(:,ind)==1);
    plot(currSpikes, ind * ones(length(currSpikes),1), '|', 'MarkerSize', 10, 'LineWidth',1.5);
end
axis('square');
xlabel('time (s)');
ylabel('neurons');
yticks([]);
xlim(tRange);
title(sprintf('Natural geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

%% Plot covariance matrices

figure('Position',[200,500,500,700],'WindowStyle','docked');
imagesc(Sigma)
colorbar;
axis('square')
title(sprintf('Target covariance, \\rho = %0.2f, p = %d, n = %d', rho, p, n))
caxis([-1,1]);
colormap(cmpBlueRed);
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
imagesc(cov(thetaNaive(maskVec,:)))
colorbar;
axis('square')
title(sprintf(' Naive geometry empirical covariance, \\rho = %0.2f, p = %d, n = %d', rho, p, n))
caxis([-1,1]);
colormap(cmpBlueRed)
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
imagesc(cov(thetaGeom(maskVec,:)))
colorbar;
axis('square')
title(sprintf('Natural geometry empirical covariance, \\rho = %0.2f, p = %d, n = %d', rho, p, n))
caxis([-1,1]);
colormap(cmpBlueRed)
ConfAxis;

%% Plot cumulative statistics after stimulus onset

cumMeanNaive = cumsum(thetaNaive(maskVec,:)) ./ (1:nnz(maskVec))';
cumMeanGeom = cumsum(thetaGeom(maskVec,:)) ./ (1:nnz(maskVec))';

cumVarNaive = cumsum(thetaNaive(maskVec,:).^2) ./  (1:nnz(maskVec))' - (cumsum(thetaNaive(maskVec,:)) ./  (1:nnz(maskVec))').^2;
cumVarGeom = cumsum(thetaGeom(maskVec,:).^2) ./  (1:nnz(maskVec))' - (cumsum(thetaGeom(maskVec,:)) ./  (1:nnz(maskVec))').^2;

tRel = tSec(maskVec) - min(tSec(maskVec));

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(interpNaive);
plot(tRel, cumMeanNaive, 'linewidth', 2);
hold on;
plot(tRel, mu(maskVec,:), '--k', 'linewidth', 2);
xlabel('time from stimulus onset (s)');
ylabel('cumulative mean')
axis('square')
title(sprintf('Naive geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(interpGeom);
plot(tRel,cumMeanGeom, 'linewidth', 2);
hold on;
plot(tRel, mu(maskVec,:), '--k', 'linewidth', 2);
xlabel('time from stimulus onset (s)');
ylabel('cumulative mean')
axis('square')
title(sprintf('Natural geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder([interpNaive; interpGeom]);
plot(tRel, [cumMeanNaive, cumMeanGeom], 'linewidth', 2);
hold on;
plot(tRel, mu(maskVec,:), '--k', 'linewidth', 2);
xlabel('time from stimulus onset (s)');
ylabel('cumulative mean')
axis('square')
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(interpNaive);
plot(tRel, cumVarNaive, 'linewidth', 2);
hold on;
plot(tRel, sigma * ones(nnz(maskVec),1), '--k', 'linewidth', 2);
xlabel('time from stimulus onset (s)');
ylabel('cumulative variance')
axis('square')
title(sprintf('Naive geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder(interpGeom);
plot(tRel,cumVarGeom, 'linewidth', 2);
hold on;
plot(tRel, sigma * ones(nnz(maskVec),1), '--k', 'linewidth', 2);
xlabel('time from stimulus onset (s)');
ylabel('cumulative variance')
axis('square')
title(sprintf('Natural geometry, \\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

figure('Position',[200,500,500,700],'WindowStyle','docked');
colororder([interpNaive; interpGeom]);
plot(tRel, [cumVarNaive, cumVarGeom], 'linewidth', 2);
hold on;
plot(tRel, sigma * ones(nnz(maskVec),1), '--k', 'linewidth', 2);
xlabel('time from stimulus onset (s)');
ylabel('cumulative variance')
axis('square')
title(sprintf('\\rho = %0.2f, p = %d, n = %d', rho, p, n));
ConfAxis;

%% Utility functions

function ConfAxis
    set(gca, 'FontSize', 16, 'LineWidth', 2, 'Box','off','TickDir','out');
end

