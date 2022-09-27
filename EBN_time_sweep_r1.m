%% Set parameters

clear variables
close all

% Number of repeats
nRep = 100;

% num parameters and num neurons
k=10;
n_n=200;
n_p=20;

% Correlation of target distribution
rho = 0.8;

% Timestep definition (seconds per step)
dt = 1e-4;

% Fix number of timesteps
T0=0.5;
T1=1.5;
tMaxSteps = round(T0/dt);

tau_m=20*(10^-3);
tau_s=0.01*tau_m;

ShortTime=(50*(10^-3));

nw2bins=200;


% Start a timer
tInit = tic;

%% control simulation
bump=true;
plots=false;
%% run the inference

% Covariance matrix of target Gaussian distribution, with trace normalized to one
Sigma=3*ones(n_p)*rho+(1-rho)*eye(n_p);

% Define sensory inputs
A0=eye(n_p,n_p);
x0=pinv(Sigma*A0')*0*ones(n_p,1);

m0 = 0;
m1=6;
Mus_0 = m0*ones(n_p,1);
Mus_1 = m1*ones(n_p,1);

if bump
    A1=eye(n_p,n_p);
    x1=pinv(Sigma*A1')*Mus_1;
else
    A1=eye(n_p,n_p);
    x1=pinv(Sigma*A1')*Mus_0;
end

%Regularization params
alpha=sqrt(n_p*k);
lambda=sqrt(n_p*k);

% Other params
S=zeros(n_p);

start_voltage=zeros(n_n,1);
start_r=zeros(n_n,1);



%% run geometry
parfor indR = 1:nRep
    
    
    W2_geo0=zeros(n_p,round((T0)/dt));
    W2_geo=zeros(n_p,round((T1)/dt));
    W2_no_geo0=zeros(n_p,round((T0)/dt));
    W2_no_geo=zeros(n_p,round((T1)/dt));

    %Decoding weights
    Gamma=randn(n_p,n_n);
    
    
    % Run the sampler
    B=sqrtm(Sigma);
    [estimate0, voltages0, rs0, os0] = EfficientBalancedSampling_classic(n_p,k,Sigma,A0,x0,Gamma,S,B,tau_m,tau_s,alpha,lambda,dt,T0,start_voltage,start_r);
    
    end_voltage=voltages0(:,end);
    end_r=rs0(:,end);
    
    [estimate1, voltages1, rs1, os1] = EfficientBalancedSampling_classic(n_p,k,Sigma,A1,x1,Gamma,S,B,tau_m,tau_s,alpha,lambda,dt,T1,end_voltage,end_r);
    
    % note that WSnorm is still zeros, working on updating to
    % faster implementation
    %                 [WSnorm0, mse0, means0, variances0] = compute_metrics_across_time(estimate0, Sigma, Mus_0, dt);
    %                 [WSnorm1, means1, variances1] = compute_metrics_across_time(estimate1, Sigma, Mus_1, dt);
    
    idx0 = ([1:round((T0)/dt)]);
    idx1 = ([1:round((T1)/dt)] + idx0(end));
    
    plot_means_geo = ((cumsum(estimate1,2)./(ones(size(estimate1,1),1)*[1:round((T1)/dt)])))';
    plot_mses_geo = (plot_means_geo - m1*ones(size(plot_means_geo))).^2;
    plot_vars_geo = ((cumsum(estimate1.^2,2) ./ (ones(size(estimate1,1),1)*[1:round((T1)/dt)]) - ((cumsum(estimate1,2) ./ (ones(size(estimate1,1),1)*[1:round((T1)/dt)])).^2))');
    
    plot_means_geo0 = ((cumsum(estimate0,2)./(ones(size(estimate0,1),1)*[1:round((T0)/dt)])))';
    plot_mses_geo0 = (plot_means_geo0 - 0*ones(size(plot_means_geo0))).^2;
    plot_vars_geo0 = ((cumsum(estimate0.^2,2) ./ (ones(size(estimate0,1),1)*[1:round((T0)/dt)]) - ((cumsum(estimate0,2) ./ (ones(size(estimate0,1),1)*[1:round((T0)/dt)])).^2))');
    
    for nt=1:round((T0)/dt)
        W2_geo0(:,nt)=EstimateW2toGaussianFromBinnedData_all(estimate0(:,1:nt)', Sigma, 0*Mus_1, nw2bins);
    end
    
    for nt=1:round((T1)/dt)
        W2_geo(:,nt)=EstimateW2toGaussianFromBinnedData_all(estimate1(:,1:nt)', Sigma, Mus_1, nw2bins);
    end
    
    Mean_Estimate_geo(indR,:)=mean([plot_means_geo0' plot_means_geo']);
    Variance_Estimate_geo(indR,:)=mean([plot_vars_geo0' plot_vars_geo']);
    W2_Estimate_geo(indR,:)=mean([W2_geo0 W2_geo]);
    
    
    %% Run with no geometry
    
    B=eye(n_p);
    
    [estimate0, voltages0, rs0, os0] = EfficientBalancedSampling_classic(n_p,k,Sigma,A0,x0,Gamma,S,B,tau_m,tau_s,alpha,lambda,dt,T0,start_voltage,start_r);
    
    end_voltage=voltages0(:,end);
    end_r=rs0(:,end);
    
    [estimate1, voltages1, rs1, os1] = EfficientBalancedSampling_classic(n_p,k,Sigma,A1,x1,Gamma,S,B,tau_m,tau_s,alpha,lambda,dt,T1,end_voltage,end_r);
    
    idx0 = ([1:round((T0)/dt)]);
    idx1 = ([1:round((T1)/dt)] + idx0(end));
    
    plot_means_no_geo = ((cumsum(estimate1,2)./(ones(size(estimate1,1),1)*[1:round((T1)/dt)])))';
    plot_mses_no_geo = (plot_means_no_geo - m1*ones(size(plot_means_no_geo))).^2;
    plot_vars_no_geo = ((cumsum(estimate1.^2,2) ./ (ones(size(estimate1,1),1)*[1:round((T1)/dt)]) - ((cumsum(estimate1,2) ./ (ones(size(estimate1,1),1)*[1:round((T1)/dt)])).^2))');
    
    plot_means_no_geo0 = ((cumsum(estimate0,2)./(ones(size(estimate0,1),1)*[1:round((T0)/dt)])))';
    plot_mses_no_geo0 = (plot_means_no_geo - m1*ones(size(plot_means_no_geo))).^2;
    plot_vars_no_geo0 = ((cumsum(estimate0.^2,2) ./ (ones(size(estimate0,1),1)*[1:round((T0)/dt)]) - ((cumsum(estimate0,2) ./ (ones(size(estimate0,1),1)*[1:round((T0)/dt)])).^2))');
    
    
    for nt=1:round((T0)/dt)
        W2_no_geo0(:,nt)=EstimateW2toGaussianFromBinnedData_all(estimate0(:,1:nt)', Sigma, 0*Mus_1, nw2bins);
    end
    
    for nt=1:round((T1)/dt)
        W2_no_geo(:,nt)=EstimateW2toGaussianFromBinnedData_all(estimate1(:,1:nt)', Sigma, Mus_1, nw2bins);
    end
    
    Mean_Estimate_no_geo(indR,:)=mean([plot_means_no_geo0' plot_means_no_geo']);
    Variance_Estimate_no_geo(indR,:)=mean([plot_vars_no_geo0' plot_vars_no_geo']);
    W2_Estimate_no_geo(indR,:)=mean([W2_no_geo0 W2_no_geo]);
    
    
    % Compute stats
    fprintf('run %d of %d \n', indR, nRep);
end



fprintf('Total Time: %f', toc(tInit));

save Mean_Estimate_geo Mean_Estimate_geo
save Variance_Estimate_geo Variance_Estimate_geo
save W2_Estimate_geo W2_Estimate_geo

save Mean_Estimate_no_geo Mean_Estimate_no_geo
save Variance_Estimate_no_geo Variance_Estimate_no_geo
save W2_Estimate_no_geo W2_Estimate_no_geo



