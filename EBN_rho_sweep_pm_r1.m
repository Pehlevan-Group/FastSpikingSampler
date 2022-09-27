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
rhos = linspace(0,0.99,100);
num_rhos = length(rhos);

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

%% control simulation
bump=true;
plots=false;

%% empty array for mean, mse, and var after tau

mean_tau_rhos_geo = zeros(n_p,num_rhos,nRep );
mean_converge_rhos_geo =zeros(n_p,num_rhos,nRep );

mse_tau_rhos_geo = zeros(n_p,num_rhos,nRep );
mse_converge_rhos_geo = zeros(n_p,num_rhos,nRep );

var_tau_rhos_geo = zeros(n_p,num_rhos,nRep );
var_converge_rhos_geo = zeros(n_p,num_rhos,nRep );

w2_tau_rhos_geo = zeros(n_p,num_rhos,nRep );
w2_converge_rhos_geo = zeros(n_p,num_rhos,nRep );

mean_tau_rhos_no_geo =zeros(n_p,num_rhos,nRep );
mean_converge_rhos_no_geo = zeros(n_p,num_rhos,nRep );
    
mse_tau_rhos_no_geo = zeros(n_p,num_rhos,nRep );
mse_converge_rhos_no_geo = zeros(n_p,num_rhos,nRep );
    
var_tau_rhos_no_geo =zeros(n_p,num_rhos,nRep );
var_converge_rhos_no_geo = zeros(n_p,num_rhos,nRep );

w2_tau_rhos_no_geo = zeros(n_p,num_rhos,nRep );
w2_converge_rhos_no_geo = zeros(n_p,num_rhos,nRep );


% Start a timer
tInit = tic;

%% run grid search across rhos, ks, and num_params
for indRho = 1:length(rhos)
   
    rho = rhos(indRho);

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

    cur_mean_tau_geo = [];
    cur_mean_converge_geo = [];

    cur_mse_tau_geo = [];
    cur_mse_converge_geo = [];

    cur_var_tau_geo = [];
    cur_var_converge_geo = [];
    
    cur_w2_tau_geo = [];
    cur_w2_converge_geo = [];
    

    cur_mean_tau_no_geo = [];
    cur_mean_converge_no_geo = [];

    cur_mse_tau_no_geo = [];
    cur_mse_converge_no_geo = [];

    cur_var_tau_no_geo = [];
    cur_var_converge_no_geo = [];
        
    cur_w2_tau_no_geo = [];
    cur_w2_converge_no_geo = [];
    
    
    %% run geometry
    for indR = 1:nRep

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

         %Current
         sample_mean_tau_geo = (plot_means_geo(round(ShortTime / dt), :));
         sample_mean_converge_geo = (plot_means_geo(end, :));


         sample_mse_tau_geo = (plot_mses_geo(round(ShortTime/ dt), :));
         sample_mse_converge_geo = (plot_mses_geo(end, :));
         
         
         sample_vars_tau_geo = (plot_vars_geo(round(ShortTime/ dt), :));
         sample_vars_converge_geo = (plot_vars_geo(end, :));
         
         sample_w2_tau_geo = EstimateW2toGaussianFromBinnedData_all(estimate1(:,1:round(ShortTime/ dt))', Sigma, Mus_1, nw2bins);
         sample_w2_converge_geo = EstimateW2toGaussianFromBinnedData_all(estimate1', Sigma, Mus_1, nw2bins);
         
         %Large array
         mean_tau_rhos_geo(:,indRho,indR) =sample_mean_tau_geo;
         mean_converge_rhos_geo(:,indRho,indR)  =sample_mean_converge_geo;
         
         mse_tau_rhos_geo(:,indRho,indR)  = sample_mse_tau_geo;
         mse_converge_rhos_geo(:,indRho,indR)  = sample_mse_converge_geo;
         
         var_tau_rhos_geo(:,indRho,indR)  =  sample_vars_tau_geo;
         var_converge_rhos_geo(:,indRho,indR)  = sample_vars_converge_geo;
         
         w2_tau_rhos_geo(:,indRho,indR)  = sample_w2_tau_geo;
         w2_converge_rhos_geo(:,indRho,indR)  =  sample_w2_converge_geo;
         
         
         % Mean only 
         cur_var_tau_geo = [cur_var_tau_geo  sample_vars_tau_geo];
         cur_mean_tau_geo = [cur_mean_tau_geo  sample_mean_tau_geo];
         cur_mse_tau_geo = [cur_mse_tau_geo  sample_mse_tau_geo];
         cur_w2_tau_geo = [cur_w2_tau_geo  sample_w2_tau_geo'];
         
         cur_w2_converge_geo = [cur_w2_converge_geo  sample_w2_converge_geo'];
         cur_mse_converge_geo = [cur_mse_converge_geo  sample_mse_converge_geo];
         cur_mean_converge_geo = [cur_mean_converge_geo  sample_mean_converge_geo];
         cur_var_converge_geo = [cur_var_converge_geo  sample_vars_converge_geo];

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

         sample_mean_tau_no_geo = (plot_means_no_geo(round(ShortTime / dt), :));
         sample_mean_converge_no_geo = (plot_means_no_geo(end, :));


         sample_mse_tau_no_geo = (plot_mses_no_geo(round(ShortTime/ dt), :));
         sample_mse_converge_no_geo = (plot_mses_no_geo(end, :));
         
         
         sample_vars_tau_no_geo = (plot_vars_no_geo(round(ShortTime/ dt), :));
         sample_vars_converge_no_geo = (plot_vars_no_geo(end, :));
         
         sample_w2_tau_no_geo = EstimateW2toGaussianFromBinnedData_all(estimate1(:,1:round(ShortTime/ dt))', Sigma, Mus_1, nw2bins);
         sample_w2_converge_no_geo = EstimateW2toGaussianFromBinnedData_all(estimate1', Sigma, Mus_1, nw2bins);
         
             %Large array
         mean_tau_rhos_no_geo(:,indRho,indR) =sample_mean_tau_no_geo;
         mean_converge_rhos_no_geo(:,indRho,indR)  =sample_mean_converge_no_geo;
         
         mse_tau_rhos_no_geo(:,indRho,indR)  = sample_mse_tau_no_geo;
         mse_converge_rhos_no_geo(:,indRho,indR)  = sample_mse_converge_no_geo;
         
         var_tau_rhos_no_geo(:,indRho,indR)  =  sample_vars_tau_no_geo;
         var_converge_rhos_no_geo(:,indRho,indR)  = sample_vars_converge_no_geo;
         
         w2_tau_rhos_no_geo(:,indRho,indR)  = sample_w2_tau_no_geo;
         w2_converge_rhos_no_geo(:,indRho,indR)  =  sample_w2_converge_no_geo;
         
         %Means
         cur_var_tau_no_geo = [cur_var_tau_no_geo  sample_vars_tau_no_geo];
         cur_mean_tau_no_geo = [cur_mean_tau_no_geo  sample_mean_tau_no_geo];
         cur_mse_tau_no_geo = [cur_mse_tau_no_geo  sample_mse_tau_no_geo];
         cur_w2_tau_no_geo = [cur_w2_tau_no_geo  sample_w2_tau_no_geo'];
         
         cur_w2_converge_no_geo = [cur_w2_converge_no_geo  sample_w2_converge_no_geo'];
         cur_mse_converge_no_geo = [cur_mse_converge_no_geo sample_mse_converge_no_geo];
         cur_mean_converge_no_geo = [cur_mean_converge_no_geo  sample_mean_converge_no_geo];
         cur_var_converge_no_geo = [cur_var_converge_no_geo  sample_vars_converge_no_geo];

    end
    
    % Compute stats
    fprintf('rho %d of %d \n', indRho, length(rhos));
% %     Means
%     mean_tau_rhos_geo(indRho)=mean(cur_mean_tau_geo);
% %     
%     mean_converge_rhos_geo(indRho)=mean(cur_mean_converge_geo);
%     
%     mean_tau_rhos_no_geo(indRho)=mean(cur_mean_tau_no_geo);
%     
%     mean_converge_rhos_no_geo(indRho)=mean(cur_mean_converge_no_geo);
%     
%     % Vars
%     var_tau_rhos_geo(indRho)=mean(cur_var_tau_geo);
%     
%     var_converge_rhos_geo(indRho)=mean(cur_var_converge_geo);
%     
%     var_tau_rhos_no_geo(indRho)=mean(cur_var_tau_no_geo);
%     
%     var_converge_rhos_no_geo(indRho)=mean(cur_var_converge_no_geo);
%     
%        % mse
%     mse_tau_rhos_geo(indRho)=mean(cur_mse_tau_geo);
%     
%     mse_converge_rhos_geo(indRho)=mean(cur_mse_converge_geo);
%     
%     mse_tau_rhos_no_geo(indRho)=mean(cur_mse_tau_no_geo);
%     
%     mse_converge_rhos_no_geo(indRho)=mean(cur_mse_converge_no_geo);
%     
%     %W2
%     w2_tau_rhos_geo(indRho)=mean(cur_w2_tau_geo);
%     
%     w2_converge_rhos_geo(indRho)=mean(cur_w2_converge_geo);
%     
%     w2_tau_rhos_no_geo(indRho)=mean(cur_w2_tau_no_geo);
%     
%     w2_converge_rhos_no_geo(indRho)=mean(cur_w2_converge_no_geo);
%     
%     
%     mean_tau_rhos_geo(:,indRho) = mean(sample_mean_tau_geo,1);
%     mean_converge_rhos_geo(indRho) = mean(sample_mean_converge_geo,1);
%     
% 
%     mse_tau_rhos_geo(indRho) = sample_mse_tau_geo / nRep;
%     mse_converge_rhos_geo(indRho) = sample_mse_converge_geo / nRep;
%     
%     var_tau_rhos_geo(indRho) = sample_var_tau_geo / nRep;
%     var_converge_rhos_geo(indRho) = sample_var_converge_geo / nRep;
% 
%     mean_tau_rhos_no_geo(indRho) = sample_mean_tau_no_geo / nRep;
%     mean_converge_rhos_no_geo(indRho) = sample_mean_converge_no_geo / nRep;
%     
%     mse_tau_rhos_no_geo(indRho) = sample_mse_tau_no_geo / nRep;
%     mse_converge_rhos_no_geo(indRho) = sample_mse_converge_no_geo / nRep;
%     
%     var_tau_rhos_no_geo(indRho) = sample_var_tau_no_geo / nRep;
%     var_converge_rhos_no_geo(indRho) = sample_var_converge_no_geo / nRep;

    clearvars -except pList kList mean_tau_rhos_geo mean_tau_rhos_no_geo mean_converge_rhos_geo mean_converge_rhos_no_geo ...
    mse_tau_rhos_geo mse_tau_rhos_no_geo mse_converge_rhos_geo mse_converge_rhos_no_geo var_tau_rhos_geo var_tau_rhos_no_geo ...
    var_converge_rhos_geo var_converge_rhos_no_geo dt tau_m tau_s indK indP indR indRho nRep tMaxSteps T0 T1 rhos bump plots ...
    n_p k n_n tInit ShortTime SEM_converge_rhos_no_geo SEM_tau_rhos_no_geo SEM_converge_rhos_geo SEM_tau_rhos_geo ...
    SEMv_converge_rhos_no_geo SEMv_tau_rhos_no_geo SEMv_converge_rhos_geo SEMv_tau_rhos_geo ...
    SEMmse_converge_rhos_no_geo SEMmse_tau_rhos_no_geo SEMmse_converge_rhos_geo SEMmse_tau_rhos_geo ...
    SEMw2_converge_rhos_no_geo SEMw2_tau_rhos_no_geo SEMw2_converge_rhos_geo SEMw2_tau_rhos_geo...
    w2_tau_rhos_geo w2_tau_rhos_no_geo w2_converge_rhos_geo w2_converge_rhos_no_geo  nw2bins
end

fprintf('Total Time: %f', toc(tInit));

%Geom
save mean_tau_rhos_geo mean_tau_rhos_geo
save mean_converge_rhos_geo mean_converge_rhos_geo



save mse_tau_rhos_geo mse_tau_rhos_geo
save mse_converge_rhos_geo mse_converge_rhos_geo



save var_tau_rhos_geo var_tau_rhos_geo
save var_converge_rhos_geo var_converge_rhos_geo



save w2_tau_rhos_geo w2_tau_rhos_geo
save w2_converge_rhos_geo w2_converge_rhos_geo


%Naive
save mean_tau_rhos_no_geo mean_tau_rhos_no_geo
save mean_converge_rhos_no_geo mean_converge_rhos_no_geo



save mse_tau_rhos_no_geo mse_tau_rhos_no_geo
save mse_converge_rhos_no_geo mse_converge_rhos_no_geo

save var_tau_rhos_no_geo var_tau_rhos_no_geo
save var_converge_rhos_no_geo var_converge_rhos_no_geo


save w2_tau_rhos_no_geo w2_tau_rhos_no_geo
save w2_converge_rhos_no_geo w2_converge_rhos_no_geo



