%% Set parameters

clear variables
close all

% Number of parameters
pList = [2, 4, 8, 16, 32, 64, 128];
num_ps = length(pList);

% Number of neurons per parameter (n = k*p)
kList = [1, 5,10, 20];
num_ks = length(kList);

% Number of repeats
nRep = 100;

% Correlation of target distribution
rho = 0.75;

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

mean_tau_params_geo = cell(num_ps, num_ks,nRep );
mean_converge_params_geo = cell(num_ps, num_ks,nRep );

mse_tau_params_geo = cell(num_ps, num_ks,nRep );
mse_converge_params_geo = cell(num_ps, num_ks,nRep );

var_tau_params_geo = cell(num_ps, num_ks,nRep );
var_converge_params_geo = cell(num_ps, num_ks,nRep );

w2_tau_params_geo = cell(num_ps, num_ks,nRep );
w2_converge_params_geo = cell(num_ps, num_ks,nRep );

mean_tau_params_no_geo = cell(num_ps, num_ks,nRep );
mean_converge_params_no_geo = cell(num_ps, num_ks,nRep );

mse_tau_params_no_geo = cell(num_ps, num_ks,nRep );
mse_converge_params_no_geo = cell(num_ps, num_ks,nRep );

var_tau_params_no_geo = cell(num_ps, num_ks,nRep );
var_converge_params_no_geo = cell(num_ps, num_ks,nRep );

w2_tau_params_no_geo = cell(num_ps, num_ks,nRep );
w2_converge_params_no_geo = cell(num_ps, num_ks,nRep );

% Start a timer
tInit = tic;

%% run grid search across rhos, ks, and num_params
for indP = 1:length(pList)
    for indK = 1:length(kList)
        
        % setup parameters
        k=kList(indK);
        n_n=pList(indP)*k;
        n_p=pList(indP);
        
        % Covariance matrix of target Gaussian distribution, with trace normalized to one
        Sigma=3*ones(n_p)*rho+(1-rho)*eye(n_p);
        
        % Define sensory inputs
        A0=eye(n_p,n_p);
        x0=pinv(Sigma*A0')*0*ones(n_p,1);
        
        m0 = 0;
        m1 = 6;
        Mus_0 = m0*ones(n_p,1);
        Mus_1 = m1*ones(n_p,1);
        
        if bump
            A1=eye(n_p,n_p);
            x1=pinv(Sigma*A1')*Mus_1;
        else
            A1=eye(n_p,n_p);
            x1=pinv(Sigma*A1')*Mus_0;
        end
        
        %Geometry
        B=sqrtm(Sigma);
        %         B=eye(n_p);
        
        
        
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
        
        for indR = 1:nRep
            
            
            %Decoding weights
            Gamma=randn(n_p,n_n);
            
            % Run the sampler
            
            B=sqrtm(Sigma);
            
            [estimate0, voltages0, rs0, os0] = EfficientBalancedSampling_classic(n_p,k,Sigma,A0,x0,Gamma,S,B,tau_m,tau_s,alpha,lambda,dt,T0,start_voltage,start_r);
            
            end_voltage=voltages0(:,end);
            end_r=rs0(:,end);
            
            [estimate1, voltages1, rs1, os1] = EfficientBalancedSampling_classic(n_p,k,Sigma,A1,x1,Gamma,S,B,tau_m,tau_s,alpha,lambda,dt,T1,end_voltage,end_r);
            
            idx0 = ([1:round((T0)/dt)]);
            idx1 = ([1:round((T1)/dt)] + idx0(end));
            
            plot_means_geo = ((cumsum(estimate1,2)./(ones(size(estimate1,1),1)*[1:round((T1)/dt)])))';
            plot_mses_geo = (plot_means_geo - m1*ones(size(plot_means_geo))).^2;
            plot_vars_geo = ((cumsum(estimate1.^2,2) ./ (ones(size(estimate1,1),1)*[1:round((T1)/dt)]) - ((cumsum(estimate1,2) ./ (ones(size(estimate1,1),1)*[1:round((T1)/dt)])).^2))');
            
            sample_mean_tau_geo = (plot_means_geo(round(ShortTime / dt), :));
            sample_mean_converge_geo = (plot_means_geo(end, :));
            
            
            sample_mse_tau_geo = (plot_mses_geo(round(ShortTime/ dt), :));
            sample_mse_converge_geo = (plot_mses_geo(end, :));
            
            
            sample_vars_tau_geo = (plot_vars_geo(round(ShortTime/ dt), :));
            sample_vars_converge_geo = (plot_vars_geo(end, :));
            
            sample_w2_tau_geo = EstimateW2toGaussianFromBinnedData_all(estimate1(:,1:round(ShortTime/ dt))', Sigma, Mus_1, nw2bins);
            sample_w2_converge_geo = EstimateW2toGaussianFromBinnedData_all(estimate1', Sigma, Mus_1, nw2bins);
            
            %Full Array
            mean_tau_params_geo{indP,indK,indR} = sample_mean_tau_geo;
            mean_converge_params_geo{indP,indK,indR} = sample_mean_converge_geo;
            
            mse_tau_params_geo{indP,indK,indR} = sample_mse_tau_geo;
            mse_converge_params_geo{indP,indK,indR} = sample_mse_converge_geo;
            
            var_tau_params_geo{indP,indK,indR} = sample_vars_tau_geo;
            var_converge_params_geo{indP,indK,indR} = sample_vars_converge_geo;
            
            w2_tau_params_geo{indP,indK,indR} = sample_w2_tau_geo;
            w2_converge_params_geo{indP,indK,indR} = sample_w2_converge_geo;
            
%             % Means
%             
%             cur_var_tau_geo = [cur_var_tau_geo  sample_vars_tau_geo];
%             cur_mean_tau_geo = [cur_mean_tau_geo  sample_mean_tau_geo];
%             cur_mse_tau_geo = [cur_mse_tau_geo  sample_mse_tau_geo];
%             cur_w2_tau_geo = [cur_w2_tau_geo  sample_w2_tau_geo'];
%             
%             cur_w2_converge_geo = [cur_w2_converge_geo  sample_w2_converge_geo'];
%             cur_mse_converge_geo = [cur_mse_converge_geo  sample_mse_converge_geo];
%             cur_mean_converge_geo = [cur_mean_converge_geo  sample_mean_converge_geo];
%             cur_var_converge_geo = [cur_var_converge_geo  sample_vars_converge_geo];
            
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
            
            %Full Array
            mean_tau_params_no_geo{indP,indK,indR} = sample_mean_tau_no_geo;
            mean_converge_params_no_geo{indP,indK,indR} = sample_mean_converge_no_geo;
            
            mse_tau_params_no_geo{indP,indK,indR} = sample_mse_tau_no_geo;
            mse_converge_params_no_geo{indP,indK,indR} = sample_mse_converge_no_geo;
            
            var_tau_params_no_geo{indP,indK,indR} = sample_vars_tau_no_geo;
            var_converge_params_no_geo{indP,indK,indR} = sample_vars_converge_no_geo;
            
            w2_tau_params_no_geo{indP,indK,indR} = sample_w2_tau_no_geo;
            w2_converge_params_no_geo{indP,indK,indR} = sample_w2_converge_no_geo;
            
%             % Mean
%             cur_var_tau_no_geo = [cur_var_tau_no_geo  sample_vars_tau_no_geo];
%             cur_mean_tau_no_geo = [cur_mean_tau_no_geo  sample_mean_tau_no_geo];
%             cur_mse_tau_no_geo = [cur_mse_tau_no_geo  sample_mse_tau_no_geo];
%             cur_w2_tau_no_geo = [cur_w2_tau_no_geo  sample_w2_tau_no_geo'];
%             
%             cur_w2_converge_no_geo = [cur_w2_converge_no_geo  sample_w2_converge_no_geo'];
%             cur_mse_converge_no_geo = [cur_mse_converge_no_geo sample_mse_converge_no_geo];
%             cur_mean_converge_no_geo = [cur_mean_converge_no_geo  sample_mean_converge_no_geo];
%             cur_var_converge_no_geo = [cur_var_converge_no_geo  sample_vars_converge_no_geo];
            
            
        end
        
        % Compute stats
        fprintf('p %d of %d, k %d of %d \n', indP, length(pList), indK, length(kList));
        
        %         %Mean
        %         mean_tau_params_geo(indP, indK)=mean(cur_mean_tau_geo);
        %         SEM_tau_params_geo(indP, indK)=std(cur_mean_tau_geo)/sqrt(n_p*nRep);
        %
        %         mean_converge_params_geo(indP, indK) =mean(cur_mean_converge_geo);
        %         SEM_converge_params_geo(indP, indK) =std(cur_mean_converge_geo/sqrt(n_p*nRep));
        %
        %
        %         mean_tau_params_no_geo(indP, indK) =mean(cur_mean_tau_no_geo);
        %         SEM_tau_params_no_geo(indP, indK) =std(cur_mean_tau_no_geo)/sqrt(n_p*nRep);
        %
        %         mean_converge_params_no_geo(indP, indK) =mean(cur_mean_converge_no_geo);
        %         SEM_converge_params_no_geo(indP, indK) =std(cur_mean_converge_no_geo)/sqrt(n_p*nRep);
        %
        %
        %
        %         %Var
        %         var_tau_params_geo(indP, indK) = mean(cur_var_tau_geo);
        %         SEMv_tau_params_geo(indP, indK)=std(cur_var_tau_geo)/sqrt(n_p*nRep);
        %
        %
        %         var_converge_params_geo(indP, indK) =mean(cur_var_converge_geo);
        %         SEMv_converge_params_geo(indP, indK) =std(cur_var_converge_geo/sqrt(n_p*nRep));
        %
        %
        %         var_tau_params_no_geo(indP, indK) =mean(cur_var_tau_no_geo);
        %         SEMv_tau_params_no_geo(indP, indK) =std(cur_var_tau_no_geo)/sqrt(n_p*nRep);
        %
        %         var_converge_params_no_geo(indP, indK) =mean(cur_var_converge_no_geo);
        %         SEMv_converge_params_no_geo(indP, indK) =std(cur_var_converge_no_geo)/sqrt(n_p*nRep);
        %
        %
        %         %MSE
        %         mse_tau_params_geo(indP, indK)=mean(cur_mse_tau_geo);
        %         SEMmse_tau_params_geo(indP, indK)=std(cur_mse_tau_geo)/sqrt(n_p*nRep);
        %
        %         mse_converge_params_geo(indP, indK)=mean(cur_mse_converge_geo);
        %         SEMmse_converge_params_geo(indP, indK) =std(cur_mse_converge_geo)/sqrt(n_p*nRep);
        %
        %         mse_tau_params_no_geo(indP, indK)=mean(cur_mse_tau_no_geo);
        %         SEMmse_tau_params_no_geo(indP, indK) =std(cur_mse_tau_no_geo)/sqrt(n_p*nRep);
        %
        %         mse_converge_params_no_geo(indP, indK)=mean(cur_mse_converge_no_geo);
        %         SEMmse_converge_params_no_geo(indP, indK) =std(cur_mse_converge_no_geo)/sqrt(n_p*nRep);
        %
        %         %W2
        %         w2_tau_params_geo(indP, indK)=mean(cur_w2_tau_geo);
        %         SEMw2_tau_params_geo(indP, indK)=std(cur_w2_tau_geo)/sqrt(n_p*nRep);
        %
        %         w2_converge_params_geo(indP, indK)=mean(cur_w2_converge_geo);
        %         SEMw2_converge_params_geo(indP, indK)=std(cur_w2_converge_geo)/sqrt(n_p*nRep);
        %
        %         w2_tau_params_no_geo(indP, indK)=mean(cur_w2_tau_no_geo);
        %         SEMw2_tau_params_no_geo(indP, indK)=std(cur_w2_tau_no_geo)/sqrt(n_p*nRep);
        %
        %         w2_converge_params_no_geo(indP, indK)=mean(cur_w2_converge_no_geo);
        %         SEMw2_converge_params_no_geo(indP, indK)=std(cur_w2_converge_no_geo)/sqrt(n_p*nRep);
        %
        
        
        clearvars -except pList kList mean_tau_params_geo mean_tau_params_no_geo mean_converge_params_geo mean_converge_params_no_geo ...
            mse_tau_params_geo mse_tau_params_no_geo mse_converge_params_geo mse_converge_params_no_geo var_tau_params_geo var_tau_params_no_geo ...
            var_converge_params_geo var_converge_params_no_geo dt tau_m tau_s indK indP indR indRho nRep tMaxSteps T0 T1 rhos bump plots ...
            n_p k n_n tInit rho ShortTime SEM_converge_params_no_geo SEM_tau_params_no_geo SEM_converge_params_geo SEM_tau_params_geo ...
            SEMv_converge_params_no_geo SEMv_tau_params_no_geo SEMv_converge_params_geo SEMv_tau_params_geo ...
            SEMmse_converge_params_no_geo SEMmse_tau_params_no_geo SEMmse_converge_params_geo SEMmse_tau_params_geo ...
            SEMw2_converge_params_no_geo SEMw2_tau_params_no_geo SEMw2_converge_params_geo SEMw2_tau_params_geo...
            w2_tau_params_geo w2_tau_params_no_geo w2_converge_params_geo w2_converge_params_no_geo  nw2bins num_ps num_ks
    end
end

fprintf('Total Time: %f', toc(tInit));


%Geom
save mean_tau_params_geo mean_tau_params_geo
save mean_converge_params_geo mean_converge_params_geo



save mse_tau_params_geo mse_tau_params_geo
save mse_converge_params_geo mse_converge_params_geo



save var_tau_params_geo var_tau_params_geo
save var_converge_params_geo var_converge_params_geo



save w2_tau_params_geo w2_tau_params_geo
save w2_converge_params_geo w2_converge_params_geo


%Naive
save mean_tau_params_no_geo mean_tau_params_no_geo
save mean_converge_params_no_geo mean_converge_params_no_geo


save mse_tau_params_no_geo mse_tau_params_no_geo
save mse_converge_params_no_geo mse_converge_params_no_geo


save var_tau_params_no_geo var_tau_params_no_geo
save var_converge_params_no_geo var_converge_params_no_geo



save w2_tau_params_no_geo w2_tau_params_no_geo
save w2_converge_params_no_geo w2_converge_params_no_geo


