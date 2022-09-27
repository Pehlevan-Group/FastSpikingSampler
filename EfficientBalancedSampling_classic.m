function [estimate, voltages, rs, os] = EfficientBalancedSampling_classic(n_p,k,Sigma,A,x,Gamma,S,B,tau_w,tau_s,alpha,lambda,dt,T,start_voltage, start_r)

n_samples=round(T/dt);

n_n=n_p*k;
n_d=n_n-n_p;

voltages=zeros(n_n,n_samples);
voltages(:,1)=start_voltage;
rs=zeros(n_n,n_samples);
rs(:,1)=start_r;
os=zeros(n_n,n_samples);

estimate=zeros(n_p,n_samples);

thresholds=diag(0.5*(lambda*eye(n_n)+Gamma'*Gamma));


D=B*B';

W=(lambda*eye(n_n)+Gamma'*Gamma);



W_Dyn=((1/tau_w*eye(n_p))-(1/tau_s*eye(n_p))*(D+S)/(Sigma))*Gamma;
% wDyn
W_in=(D+S)*A';
W_noise=((Gamma'*B));

for t=2:n_samples

    Dynamics=W_Dyn*rs(:,t-1)+(1/tau_s)*W_in*x;
    
    
    dV=dt*(-(1/tau_w)*(voltages(:,t-1)+alpha)+Gamma'*Dynamics)-W*os(:,t-1)+W_noise*sqrt(dt*(2)/tau_s)*randn(n_p,1);
    voltages(:,t)=voltages(:,t-1)+dV;
    
    [maxval,idx]=max(voltages(:,t));%+randn(n_n,1));
    
    if maxval>thresholds(idx)
        os(idx,t)=1;
    end
    
%     if t>round(0.001/dt)
%         idx=voltages(:,t)>thresholds & sum(os(:,t-round(0.001/dt):t),2)==0;
%     else
%         idx=voltages(:,t)>thresholds & sum(os(:,1:t),2)==0;
%     end
%     os(idx,t)=1;
    
    rs(:,t) = rs(:,t-1) + dt * ((-1 / tau_w) * rs(:,t-1)) + os(:,t);
    
    estimate(:,t)=Gamma(1:n_p,:)*rs(:,t);
    
end




%Implements sampling in efficient balanced networks
%   Detailed explanation goes here







end

