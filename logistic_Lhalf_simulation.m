clear
clc
%rng(1);

%%%%%%%%%% generate benchmark dataset %%%%%%%%%%%%%%%%%%%%%%%%
beta=zeros(1,1000);
beta(1)=5;
beta(2)=-5;
beta(3)=5;
beta(4)=-5;
beta(5)=5;
beta(6)=-5;
beta(7)=5;
beta(8)=-5;
beta(9)=5;
beta(10)=-5;

% beta(1)=1.2;
% beta(4)=1.6;
% beta(7)=0.9;
% beta(15)=0.6;
% beta(19)=0.5;
% beta(23)=-1.2;
% beta(26)=1;
% beta(30)=-0.5;
% beta(35)=1.3;
% beta(36)=0.8;

actual_beta=beta;

train_size=100;
test_size=50;
sample_size=train_size+test_size;

intercept=0.0;
X = normrnd(0, 1, sample_size, size(beta,2)+1);
[n,p]=size(X);
cor=0.0;
for i=1:n
    for j=1:p-1
        x(i,j)=X(i,j+1)*sqrt(1-cor)+X(i,1)*sqrt(cor);
    end
end

l = intercept + x * beta'; %l=intercept+(x*beta'+ 0.2*normrnd(0, 1, n, 1));
prob=exp(l)./(1 + exp(l));
U=rand(1,sample_size);

for i=1:sample_size
    if prob(i)>0.5
        y(i)=1;
    else
        y(i)=0;
    end
end
y=y';

x_test=x(train_size+1:sample_size,:);
x=x(1:train_size,:);
y_test=y(train_size+1:sample_size,:);
y=y(1:train_size,:);

col=size(x,2);
row=size(x,1);

temp=sum(y)/row;
beta_zero=log(temp/(1-temp));    %intercept
beta=zeros(col,1);

%%%%%%%%%%%%%% compute lambda on the log scale %%%%%%%%%%%%%%%%%%%%%
eta = beta_zero + x * beta;
Pi=exp(eta)./(1+exp(eta));
W=diag(Pi.*(1-Pi));           %%%%%%%%% W is diagonal matrix%%%%%%%%%%
r=(W^-1)*(y-Pi);              % residual
S=(x'*W*r)/row;
lambda_max=(4/3*(max(S)))^(1.5);
lambda_min = lambda_max*0.001;
m = 10;

for i=1:m
    Lambda(i) = lambda_max*(lambda_min/lambda_max)^(i/m);
end
for i=1:m    
   lambda=Lambda(i);
   iter=0;
   maxiter=1000;
   beta_path(:,i)=beta(:,1);
   
   while iter<maxiter
    
    beta_temp=beta;
    beta_zero_temp=beta_zero;
    
    eta = beta_zero_temp + x * beta_temp;   
    Pi=exp(eta)./(1+exp(eta));
    W=diag(Pi.*(1-Pi));         
    [m,n]=size(W);
    r=(W^-1)*(y-Pi);      
    
    %%%%%%%%%%%%%%%%%%%% initializing beta_0 and beta_k %%%%%%%%%%%%%%%%%%%%%%
    beta_zero=sum(W*r)/sum(sum(W))+beta_zero_temp;    
    r=r-(beta_zero-beta_zero_temp);                
    
    for j=1:col
        v=x(:,j)'*W*x(:,j)/row;
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
    iter=iter+1
   end    
end
[opt,Mse]=CV_logistic(x,y,Lambda,beta_path);

beta=beta_path(:,opt);
l= intercept + x_test * beta;
prob=exp(l)./(1 + exp(l));
for i=1:test_size
    if prob(i)>0.5
        test_y(i)=1;
    else
        test_y(i)=0;
    end
end
error=test_y'-y_test;
count=find(error~=0)
fail=length(count)

beta_non_zero=find(beta~=0);

plot(beta_path','linewidth',1.5)
ax = axis;
line([opt opt], [ax(3) ax(4)], 'Color', 'b', 'LineStyle', '-.');
xlabel('Steps')
ylabel('Coefficeints')

figure;
hold on
plot(Mse,'linewidth',1.5);
ax = axis;
line([opt opt], [ax(3) ax(4)], 'Color', 'b', 'LineStyle', '-.');
xlabel('Steps')
ylabel('Misclassification Error')
