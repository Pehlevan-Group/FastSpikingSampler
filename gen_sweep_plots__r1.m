clear variables;
close all;


%% Load in the data

all_files=dir(fullfile( '*.mat'));

for file = all_files'
    load(fullfile(file.name));
end
% 
% 
% rho_files = dir(fullfile('rho_sweep', '*.mat'));
% for file = rho_files'
%     load(fullfile('rho_sweep/', file.name));
% end
% % 
% % param_files = dir(fullfile('param_sweep', '*.mat'));
% % for file = param_files'
% %     load(fullfile('param_sweep/', file.name))
% % end

corder = [0.850980392156863, 0.372549019607843, 0.007843137254902; 0.458823529411765, 0.439215686274510, 0.701960784313725];

%% generate the rho sweep plots

rhos = linspace(0, 0.99, 100);
pList = [2, 4, 8, 16, 32, 64, 128];
kList = [1, 5,10, 20];
dt=1e-4;
nBoot=1000;
nRep=100;
%After 50ms


%% rho sweep
% mean rho
F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
MeanDim_tau_rhos_no_geo=squeeze(mean(mean_tau_rhos_no_geo,1));
MeanDim_tau_rhos_geo=squeeze(mean(mean_tau_rhos_geo,1));
x = [squeeze(mean(MeanDim_tau_rhos_no_geo,2))'; squeeze(mean(MeanDim_tau_rhos_geo,2))'];
ciNaive = bootci(nBoot, @mean, MeanDim_tau_rhos_no_geo')';
ciGeom = bootci(nBoot, @mean, MeanDim_tau_rhos_geo')';
PlotAsymmetricErrorPatch(rhos', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on 
plot([0 1],[6 6],'k')
% legend('Naive', 'Geometry')
axis square
% title('Mean at T=tau_m')
xlabel('\rho')
ylabel('Mean')
ax=FormatAxis;
ax.FontName='arial';
ax.YLim=[min(x(:))-0.1 6.5];
ax.XTick=[0 0.25 0.5 0.75 1];
ax.YTick=[0 3 6];
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'meantaurhos','pdf')

% Var rho
F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
MeanVar_tau_rhos_no_geo=squeeze(mean(var_tau_rhos_no_geo,1));
MeanVar_tau_rhos_geo=squeeze(mean(var_tau_rhos_geo,1));
x = [squeeze(mean(MeanVar_tau_rhos_no_geo,2))'; squeeze(mean(MeanVar_tau_rhos_geo,2))'];
ciNaive = bootci(nBoot, @mean, MeanVar_tau_rhos_no_geo')';
ciGeom = bootci(nBoot, @mean, MeanVar_tau_rhos_geo')';
PlotAsymmetricErrorPatch(rhos', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on
plot([0 1],[3 3],'k')
% legend('Naive', 'Geometry')
% title('Variance at T=tau_m')
xlabel('\rho')
ylabel('Variance')
ax=FormatAxis;
ax.FontName='arial';
ax.YLim=[0 max(x(:))+0.1 ];
ax.XTick=[0 0.25 0.5 0.75 1];
ax.YTick=[0 5 10 15];
axis square
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'vartaurhos','pdf')


% W2 rho
F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
Meanw2_tau_rhos_no_geo=squeeze(mean(w2_tau_rhos_no_geo,1));
Meanw2_tau_rhos_geo=squeeze(mean(w2_tau_rhos_geo,1));
x = [squeeze(mean(Meanw2_tau_rhos_no_geo,2))'; squeeze(mean(Meanw2_tau_rhos_geo,2))'];
ciNaive = bootci(nBoot, @mean, Meanw2_tau_rhos_no_geo')';
ciGeom = bootci(nBoot, @mean, Meanw2_tau_rhos_geo')';
PlotAsymmetricErrorPatch(rhos', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
% title('W2 distance at T=tau_m')
xlabel('\rho')
ylabel('W2 distance')
ax=FormatAxis;
ax.FontName='arial';

ax.YLim=[0 max(x(:))+0.1 ];
ax.XTick=[0 0.25 0.5 0.75 1];
ax.YTick=[0 25 50];
axis square
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'w2taurhos','pdf')

% set(gca, 'YScale', 'log')



%% parameters sweep


np=[2,4,8,16,32,64];

%Only plotting for k=10 here
TotalMean_no_geo=zeros(length(np),nRep);
TotalMean_geo=zeros(length(np),nRep);

TotalVar_no_geo=zeros(length(np),nRep);
TotalVar_geo=zeros(length(np),nRep);

TotalW2_no_geo=zeros(length(np),nRep);
TotalW2_geo=zeros(length(np),nRep);


for n=1:length(np)
    CurrentMean_no_geo=zeros(nRep,1);
    CurrentMean_geo=zeros(nRep,1);
    CurrentVar_no_geo=zeros(nRep,1);
    CurrentVar_geo=zeros(nRep,1);
    CurrentW2_no_geo=zeros(nRep,1);
    CurrentW2_geo=zeros(nRep,1);
    
    for nr=1:nRep
        CurrentMean_no_geo(nr)=mean(mean_tau_params_no_geo{n,3,nr});
        CurrentMean_geo(nr)=mean(mean_tau_params_geo{n,3,nr});
        CurrentVar_no_geo(nr)=mean(var_tau_params_no_geo{n,3,nr});
        CurrentVar_geo(nr)=mean(var_tau_params_geo{n,3,nr});
        CurrentW2_no_geo(nr)=mean(w2_tau_params_no_geo{n,3,nr});
        CurrentW2_geo(nr)=mean(w2_tau_params_geo{n,3,nr});
    end
    TotalMean_no_geo(n,:)=CurrentMean_no_geo;
    TotalMean_geo(n,:)=CurrentMean_geo;
    
    TotalVar_no_geo(n,:)=CurrentVar_no_geo;
    TotalVar_geo(n,:)=CurrentVar_geo;
    
    TotalW2_no_geo(n,:)=CurrentW2_no_geo;
    TotalW2_geo(n,:)=CurrentW2_geo;
end

% Mean params
F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
X1=mean(TotalMean_no_geo,2);
X2=mean(TotalMean_geo,2);
x = [X1';X2'];
ciNaive = bootci(nBoot, @mean, TotalMean_no_geo')';
ciGeom = bootci(nBoot, @mean, TotalMean_geo')';
PlotAsymmetricErrorPatch(np', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on 
plot([1 65],[6 6],'k')
% legend('Naive', 'Geometry')
% title('Mean at T=tau_m')
xlabel('n_p')
ylabel('Mean')
ax=FormatAxis;
ax.FontName='arial';
ax.XLim=[2 65];
ax.YLim=[0 6.2];
ax.YTick=[0 3 6];
ax.XTick=[2 4 8 16 32 64];
set(gca, 'XScale', 'log')
axis square
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'meantauparams','pdf')



F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
X1=mean(TotalVar_no_geo,2);
X2=mean(TotalVar_geo,2);
x = [X1';X2'];
ciNaive = bootci(nBoot, @mean, TotalVar_no_geo')';
ciGeom = bootci(nBoot, @mean, TotalVar_geo')';
PlotAsymmetricErrorPatch(np', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on 
plot([1 65],[3 3],'k')
xlabel('n_p')
ylabel('Variance')
ax=FormatAxis;
ax.FontName='arial';
ax.XLim=[2 65];
ax.YTick=[0 5 10 15];
ax.XTick=[2 4 8 16 32 64];
set(gca, 'XScale', 'log')
axis square
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'vartauparams','pdf')




F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
X1=mean(TotalW2_no_geo,2);
X2=mean(TotalW2_geo,2);
x = [X1';X2'];
ciNaive = bootci(nBoot, @mean, TotalW2_no_geo')';
ciGeom = bootci(nBoot, @mean, TotalW2_geo')';
PlotAsymmetricErrorPatch(np', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on 
% title('W2 distance at T=tau_m')
xlabel('n_p')
ylabel('W2 distance')
ax=FormatAxis;
ax.FontName='arial';
ax.XTick=[2 4 8 16 32 64];
ax.XLim=[2 65];
axis square
set(gca, 'XScale', 'log')
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'W2tauparams','pdf')

%% At steady state


%% rho sweep
% mean rho
F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
MeanDim_converge_rhos_no_geo=squeeze(mean(mean_converge_rhos_no_geo,1));
MeanDim_converge_rhos_geo=squeeze(mean(mean_converge_rhos_geo,1));
x = [squeeze(mean(MeanDim_converge_rhos_no_geo,2))'; squeeze(mean(MeanDim_converge_rhos_geo,2))'];
ciNaive = bootci(nBoot, @mean, MeanDim_converge_rhos_no_geo')';
ciGeom = bootci(nBoot, @mean, MeanDim_converge_rhos_geo')';
PlotAsymmetricErrorPatch(rhos', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on 
plot([0 1],[6 6],'k')
% legend('Naive', 'Geometry')
axis square
% title('Mean at T=converge_m')
xlabel('\rho')
ylabel('Mean')
ax=FormatAxis;
ax.FontName='arial';
ax.YLim=[min(x(:))-0.1 6.5];
ax.XTick=[0 0.25 0.5 0.75 1];
ax.YTick=[0 3 6];
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'meanconvergerhos','pdf')

% Var rho
F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
MeanVar_converge_rhos_no_geo=squeeze(mean(var_converge_rhos_no_geo,1));
MeanVar_converge_rhos_geo=squeeze(mean(var_converge_rhos_geo,1));
x = [squeeze(mean(MeanVar_converge_rhos_no_geo,2))'; squeeze(mean(MeanVar_converge_rhos_geo,2))'];
ciNaive = bootci(nBoot, @mean, MeanVar_converge_rhos_no_geo')';
ciGeom = bootci(nBoot, @mean, MeanVar_converge_rhos_geo')';
PlotAsymmetricErrorPatch(rhos', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on
plot([0 1],[3 3],'k')
% legend('Naive', 'Geometry')
% title('Variance at T=converge_m')
xlabel('\rho')
ylabel('Variance')
ax=FormatAxis;
ax.FontName='arial';
ax.YLim=[0 max(x(:))+0.1 ];
ax.XTick=[0 0.25 0.5 0.75 1];
ax.YTick=[0 5 10 15];
axis square
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'varconvergerhos','pdf')


% W2 rho
F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
Meanw2_converge_rhos_no_geo=squeeze(mean(w2_converge_rhos_no_geo,1));
Meanw2_converge_rhos_geo=squeeze(mean(w2_converge_rhos_geo,1));
x = [squeeze(mean(Meanw2_converge_rhos_no_geo,2))'; squeeze(mean(Meanw2_converge_rhos_geo,2))'];
ciNaive = bootci(nBoot, @mean, Meanw2_converge_rhos_no_geo')';
ciGeom = bootci(nBoot, @mean, Meanw2_converge_rhos_geo')';
PlotAsymmetricErrorPatch(rhos', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
% title('W2 distance at T=converge_m')
xlabel('\rho')
ylabel('W2 distance')
ax=FormatAxis;
ax.FontName='arial';

ax.YLim=[0 max(x(:))+0.1 ];
ax.XTick=[0 0.25 0.5 0.75 1];
ax.YTick=[0 25 50];
axis square
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'w2convergerhos','pdf')

% set(gca, 'YScale', 'log')



%% parameters sweep


np=[2,4,8,16,32,64];

%Only plotting for k=10 here
TotalMean_no_geo=zeros(length(np),nRep);
TotalMean_geo=zeros(length(np),nRep);

TotalVar_no_geo=zeros(length(np),nRep);
TotalVar_geo=zeros(length(np),nRep);

TotalW2_no_geo=zeros(length(np),nRep);
TotalW2_geo=zeros(length(np),nRep);


for n=1:length(np)
    CurrentMean_no_geo=zeros(nRep,1);
    CurrentMean_geo=zeros(nRep,1);
    CurrentVar_no_geo=zeros(nRep,1);
    CurrentVar_geo=zeros(nRep,1);
    CurrentW2_no_geo=zeros(nRep,1);
    CurrentW2_geo=zeros(nRep,1);
    
    for nr=1:nRep
        CurrentMean_no_geo(nr)=mean(mean_converge_params_no_geo{n,3,nr});
        CurrentMean_geo(nr)=mean(mean_converge_params_geo{n,3,nr});
        CurrentVar_no_geo(nr)=mean(var_converge_params_no_geo{n,3,nr});
        CurrentVar_geo(nr)=mean(var_converge_params_geo{n,3,nr});
        CurrentW2_no_geo(nr)=mean(w2_converge_params_no_geo{n,3,nr});
        CurrentW2_geo(nr)=mean(w2_converge_params_geo{n,3,nr});
    end
    TotalMean_no_geo(n,:)=CurrentMean_no_geo;
    TotalMean_geo(n,:)=CurrentMean_geo;
    
    TotalVar_no_geo(n,:)=CurrentVar_no_geo;
    TotalVar_geo(n,:)=CurrentVar_geo;
    
    TotalW2_no_geo(n,:)=CurrentW2_no_geo;
    TotalW2_geo(n,:)=CurrentW2_geo;
end

% Mean params
F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
X1=mean(TotalMean_no_geo,2);
X2=mean(TotalMean_geo,2);
x = [X1';X2'];
ciNaive = bootci(nBoot, @mean, TotalMean_no_geo')';
ciGeom = bootci(nBoot, @mean, TotalMean_geo')';
PlotAsymmetricErrorPatch(np', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on 
plot([1 65],[6 6],'k')
% legend('Naive', 'Geometry')
% title('Mean at T=converge_m')
xlabel('n_p')
ylabel('Mean')
ax=FormatAxis;
ax.FontName='arial';
ax.XLim=[2 65];
ax.YLim=[0 6.2];
ax.YTick=[0 3 6];
ax.XTick=[2 4 8 16 32 64];
set(gca, 'XScale', 'log')
axis square
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'meanconvergeparams','pdf')



F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
X1=mean(TotalVar_no_geo,2);
X2=mean(TotalVar_geo,2);
x = [X1';X2'];
ciNaive = bootci(nBoot, @mean, TotalVar_no_geo')';
ciGeom = bootci(nBoot, @mean, TotalVar_geo')';
PlotAsymmetricErrorPatch(np', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on 
plot([1 65],[3 3],'k')
xlabel('n_p')
ylabel('Variance')
ax=FormatAxis;
ax.FontName='arial';
ax.XLim=[2 65];
ax.YTick=[0 5 10 15];
ax.XTick=[2 4 8 16 32 64];
set(gca, 'XScale', 'log')
axis square
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'varconvergeparams','pdf')




F=figure('Position',[200,500,500,700],'WindowStyle','docked'); 
X1=mean(TotalW2_no_geo,2);
X2=mean(TotalW2_geo,2);
x = [X1';X2'];
ciNaive = bootci(nBoot, @mean, TotalW2_no_geo')';
ciGeom = bootci(nBoot, @mean, TotalW2_geo')';
PlotAsymmetricErrorPatch(np', x', [ciNaive(:,1), ciGeom(:,1)], [ciNaive(:,2), ciGeom(:,2)], corder);
hold on 
% title('W2 distance at T=converge_m')
xlabel('n_p')
ylabel('W2 distance')
ax=FormatAxis;
ax.FontName='arial';
ax.XTick=[2 4 8 16 32 64];
ax.XLim=[2 65];
axis square
set(gca, 'XScale', 'log')
ConfAxis;
print(F,cd, '-dpdf','-painters', '-bestfit');
saveas(F,'W2convergeparams','pdf')





%% At steady state
%


%% Utility functions

function ConfAxis
    set(gca, 'FontSize', 30,'FontName','arial', 'LineWidth', 2, 'Box','off','TickDir','out');
end