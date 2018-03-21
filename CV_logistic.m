function [ opt_s,Mse ] = CV_logistic(X,y,Lambda,beta_path)

%%%%%%%%%%%%%%     K cross validation    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=5;            %%% K-fold%%%
[n,p] = size(X);
valida_n=floor(n/k);
sample_sequence=1:n;

for j=1:length(Lambda)
    lambda=Lambda(j);
    beta_ini=beta_path(:,j);
    for i=1:k
        if i<=k-1
            validation_seq=sample_sequence(:,(i-1)*valida_n+1:i*valida_n);
        else
            validation_seq=sample_sequence(:,(i-1)*valida_n+1:n);
        end
        train_seq=setdiff(sample_sequence,validation_seq);
        X_train = X(train_seq,:);
        y_train = y(train_seq);
        X_validation= X(validation_seq, :);
        y_validation = y(validation_seq);
        [b,intercept]=Lhalf_CD_logistic(X_train,y_train,lambda,beta_ini);
        test_y=intercept+X_validation*b;
        for m=1:size(y_validation,1)
            if sign(test_y(m))==1
                test_y(m)=1;
            else
                test_y(m)=0;
            end
        end
        error=test_y-y_validation;  
        Mse(i,j)=sum(abs(error));
        test_y=0;
    end
end
[d,opt_s]=min(sum(Mse,1));
Mse=min(sum(Mse,1));

end


