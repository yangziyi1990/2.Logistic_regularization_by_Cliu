clear
clc

% load x
% load y

%%%% input training dataset and testing dataset %%%%
%x = x;  
%y = y;
%test_data = x;
%test_label = y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_size=size(x,1);
test_size=size(test_data,1);
sample_size=train_size+test_size;

col=size(x,2);
row=size(x,1);

temp=sum(y)/row;
beta_zero=log(temp/(1-temp));    %intercept
beta=zeros(col,1);

%%%%%%%%%%%%%% compute lambda on the log scale %%%%%%%%%%%%%%%%%%%%%
eta = beta_zero + x * beta;
Pi=exp(eta)./(1+exp(eta));
W=diag(Pi.*(1-Pi));           %%%%%%%%% W is diagonal matrix%%%%%%%%%%
r=(W^-1)*(y-Pi);              %residual
S=(x'*W*r)/row;
lambda_max=(4/3*(max(S)))^(1.5);
lambda_min = lambda_max*0.001;
m = 10;

for i=1:m
    Lambda(i) = lambda_max*(lambda_min/lambda_max)^(i/m);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:m
    
   lambda=Lambda(i);
   iter=0;
   maxiter=1000;
   beta_path(:,i)=beta(:,1);   
   while iter<maxiter
       beta_temp=beta;
       beta_zero_temp=beta_zero;
       
       eta=beta_zero_temp+x*beta_temp;   %%%%% eta= intercept + X*beta;
       Pi=exp(eta)./(1+exp(eta));
       W=diag(Pi.*(1-Pi));           %%%%%%%%% W is diagonal matrix%%%%%%%%%%
       [m,n]=size(W);
       r=(W^-1)*(y-Pi);            %residual= (w^-1)*(y-pi)
    
    %%%%%%%%%%%%%%%%%%%% intercept%%%%%%%%%%%%%%%%%%%%%%
        beta_zero=sum(W*r)/sum(sum(W))+beta_zero_temp;    
        r=r-(beta_zero-beta_zero_temp);                
    
        for j=1:col
            v=x(:,j)'*W*x(:,j)/row;
            S=(x(:,j)'*W*r)/row+beta_temp*v;       
      %%%%%%%%%%%%%%%%%% MCP Thresholding(Zhang,et al 2010) %%%%%%%%%%%%%%%%%
      gamma=19;%20;
      if abs(S(j))<= gamma*lambda
          if S(j) > lambda
              beta(j) = (S(j) - lambda)/((1-1/gamma));
          elseif S(j) < -lambda
              beta(j) = (S(j) + lambda)/((1-1/gamma));
          elseif abs(S(j)) <= lambda
              beta(j) = 0;
          end
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

[opt,Mse]=CV_MCP_logistic(x,y,Lambda,beta_path);

beta=beta_path(:,opt);
l=intercept+x_test*beta;
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
