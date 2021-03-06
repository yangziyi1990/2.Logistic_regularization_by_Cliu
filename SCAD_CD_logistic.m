function [beta,intercept] = SCAD_CD_logistic(x,y,lambda,beta_ini)
%LOGISTIC Summary of this function goes here
%   Detailed explanation goes here

col=size(x,2);
row=size(x,1);
temp=sum(y)/row;
beta_zero=log(temp/(1-temp));   
beta=beta_ini;

iter=0;
maxiter=10;
while iter<maxiter 
    
    beta_temp=beta;
    beta_zero_temp=beta_zero;
    
    eta=beta_zero_temp+x*beta_temp;  
    Pi=exp(eta)./(1+exp(eta));
    W=diag(Pi.*(1-Pi));           %%%%%%%%% W is diagonal matrix%%%%%%%%%%
    r=(W^-1)*(y-Pi);              %residual
    
    %%%%%%%%%%%%%%%%%%%% intercept%%%%%%%%%%%%%%%%%%%%%%
    beta_zero=sum(W*r)/sum(sum(W))+beta_zero_temp;
    intercept=beta_zero;
    r=r-(beta_zero-beta_zero_temp);           
    
    for j=1:col
        v=x(:,j)'*W*x(:,j)/row;
        S=(x(:,j)'*W*r)/row+beta_temp*v;        
        
        %%%%%%%%%%%%%%%%%%%% SCAD Thresholding (FAN & Li, 2001)%%%%%%%%%%%%%%%%
        alpha=3.7; 
        s=0;
        if S(j)>0
            s=1;
        elseif S(j)<0
            s=-1;       
        end
        if abs(S(j)) <=lambda 
            beta(j)=0;
        elseif abs(S(j)) <= 2*lambda 
            beta(j)=s*(abs(S(j))-lambda);    
        elseif abs(S(j)) <= alpha*lambda
            beta(j)=s*(abs(S(j))-alpha*lambda/(alpha-1))/(1-1/(alpha-1));
        else
            beta(j)=S(j);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%--update r---%%%%%%%%%%%%%%%%%%%%%%%%%%
        r=r-x(:,j)*(beta(j)-beta_temp(j));
        
    end    
    if norm(beta_temp - beta) < (1E-5)
        break;
    end    
    iter=iter+1;    
end

end
