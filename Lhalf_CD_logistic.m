function [beta,intercept] = Lhalf_CD_logistic(x,y,lambda,beta_ini)
%LOGISTIC Summary of this function goes here
%   Detailed explanation goes here

col=size(x,2);
row=size(x,1);

temp=sum(y)/row;
beta_zero=log(temp/(1-temp));    
beta=beta_ini;

iter=0;
maxiter=1;
while iter<maxiter     
    beta_temp=beta;
    beta_zero_temp=beta_zero;    
    eta=beta_zero_temp+x*beta_temp;   
    Pi=exp(eta)./(1+exp(eta));
    W=diag(Pi.*(1-Pi));        
    r=(W^-1)*(y-Pi);         
    %%%%%%%%%%%%%%%%%%%% intercept%%%%%%%%%%%%%%%%%%%%%%
    beta_zero=sum(W*r)/sum(sum(W))+beta_zero_temp;
    intercept=beta_zero;
    r=r-(beta_zero-beta_zero_temp);            
    
    for j=1:col
        v=x(:,j)'*W*x(:,j)/row;
        v=1;
        S=(x(:,j)'*W*r)/row+beta_temp*v;        
    %%%%%%%%%%%%%%%%%% Half Thresholding(Xu,et al 2010) %%%%%%%%%%%%%%%%%
         
        if abs(S(j)) > ((3/4)*(lambda^(2/3)))
            phi = acos(lambda/8*((abs(S(j))/3)^(-1.5)));
            beta(j) = real(2/3*S(j)*(1 + cos(2/3*(pi - phi))))/v; 
        else
            beta(j) = 0;
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


