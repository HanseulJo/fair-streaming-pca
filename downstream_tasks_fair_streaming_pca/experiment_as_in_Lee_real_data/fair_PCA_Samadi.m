% Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0


% % names = {'COMPAS', 'German', 'Adult'};
names = {'German'};

for embedding_dim=[2,10]
    for name_num = 1:length(names)
        for split = 1:10

            fprintf('Embedding dim = %d, dataset = %s, split = %d', embedding_dim, names{name_num}, split-1);

            %% Load datas
            X_train = table2array(readtable(sprintf('fair-manifold-pca/datasets/%s/train_%d.csv', names{name_num}, split-1)));
            Y_train = X_train(:, end-1);
            Z_train = X_train(:, end);
            X_train = X_train(:, 1:end-2);

            X_test = readmatrix(sprintf('fair-manifold-pca/datasets/%s/test_%d.csv', names{name_num}, split-1));
            X_test = X_test(:, 1:end-2);         

            % centering the data and normalizing the variance across each column
            for i=1:size(X_train,2)
                temp_mean = mean(X_train(:,i));
                X_train(:,i) = X_train(:,i) - temp_mean;
                X_test(:,i) = X_test(:,i) - temp_mean;
                temp_std = std(X_train(:,i));
                if temp_std > 0.0001
                    X_train(:,i) = X_train(:,i) / temp_std;
                    X_test(:,i) = X_test(:,i) / temp_std;
                end
            end
            
            writematrix(X_test,sprintf('fair-manifold-pca/uci/%s/PCA_Samadi_original_TEST_%d_dim%d.csv',names{name_num}, split-1, embedding_dim))
            
            A = X_train(logical(Z_train),:);
            A = normalize(A, 'center');
            B = X_train(~logical(Z_train),:);
            B = normalize(B, 'center');



            % THE FOLLOWING PART IS TAKEN FROM https://github.com/samirasamadi/Fair-PCA/blob/master/main_re.m

            % parameters of the mw algorithm
            eta = 20;
            T = 5; 

            %Fair PCA part
            [P_fair, z, P_last, z_last] = mw(A, B, embedding_dim, eta/embedding_dim, T);
            if z < z_last
                P_smart = P_fair;
            else
                P_smart = P_last;
            end
            P_smart = eye(size(P_smart,1)) - sqrtm(eye(size(P_smart,1))-P_smart);
            P_smart = real(P_smart);
            
            approxFair = X_train * P_smart;
            approxFair_test = X_test * P_smart;
            
            %find out the effective dimension of the embedding
            [coeff,score,latent] = pca(approxFair);
            effective_dimension = sum(latent>0.001);
            disp('Effective dimension:')
            disp(effective_dimension)
            
            writematrix(approxFair_test,sprintf('fair-manifold-pca/uci/%s/PCA_Samadi_embedding_TEST_%d_dim%d.csv',names{name_num}, split-1, embedding_dim))
        end
    end
end

function [P, z, P_last, z_last] = mw(A, B, d, eta, T)

    % matrix A has the points in group A as its rows
    % matrix B has the points in group B as its rows
    % population A and B are expected to be normalized to have mean 0. 
    % d is the target dimension
    % eta and T are MW's parameters

    disp('MW method is called')
    
    covA = transpose(A)*A;
    covB = transpose(B)*B;
    
    % m_A and m_B are size of data set A and B respectively
    m_A = size(A,1);
    m_B = size(B,1);
    n = size(A, 2);
    
    Ahat = optApprox(A, d);
    alpha = norm(Ahat, 'fro')^2;
    
    Bhat = optApprox(B, d);
    beta = norm(Bhat, 'fro')^2;
    
    % MW
    
    % start with uniform weight
    w_1 = 0.5;
    w_2 = 0.5;
    
    % P is our answer, so I keep the sum of all P_t along the way
    P = zeros(n);
    %just for record at the end to see the progress over iterates
    record = ["iteration" "w_1" "w_2" "loss A" "loss B" "loss A by average" "loss B by average"];
    
    for t=1:T
      
        %think of P_temp as P_t we got by weighting with w_1,w_2
        [P_temp,z_1,z_2] = oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2);
        
        %z_1, z_2 are losses for group A and B respectively. If z_i is big, group i is
        %bottle neck, so weight group i more next time
        w_1star = w_1*exp(eta*z_1);
        w_2star = w_2*exp(eta*z_2);
       
        %renormalize
        w_1 = w_1star / (w_1star+w_2star);
        w_2 = w_2star / (w_1star+w_2star);
        
        %add to the sum of P_t
        P = P+P_temp;
        
        %record the progress
        P_average = (1/t).*P;
        record = [record; t w_1 w_2 z_1 z_2 (1/m_A)*(alpha - sum(sum( covA .* P_average ))) (1/m_B)*(beta - sum(sum( covB .* P_average )))];
    end
    
    %take average of P_t
    P = (1/T).*P;
    
    %calculate loss of P_average
    z_1 = 1/(m_A)*(alpha - sum(sum(covA.*P)));
    z_2 = 1/(m_B)*(beta - sum(sum(covB.*P)));
    z = max(z_1,z_2);
    
    %in case last iterate is preferred to the average
    P_last = P_temp;
    
    %calculate loss of P_average
    zl_1 = 1/(m_A)*(alpha - sum(sum(covA.*P_last)));
    zl_2 = 1/(m_B)*(beta - sum(sum(covB.*P_last)));
    z_last = max(zl_1,zl_2);
    
    disp(['MW method is finished. The loss for group A is ',num2str(z_1),'For group B is ',num2str(z_2)])
    disp(record)
end


function [P_o, z_1, z_2] = oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2)

    %Given an input matrix_A= A^T A, matrix_B=B^T B both of size n by n, d, and weights w_1,w_2, solve the
    %optimization problem
    % min w_1 z_1 + w_2 z_2 s.t.
    % z_1 >= alpha - <matrix_A , P>
    % z_2 >= beta - <matrix_B , P>
    % tr(P) <= d
    % 0 <= P <= I
    
    if size(A) ~= [m_A,n] | size(B) ~= [m_B,n] %wrong size
        error('Input matrix to oracle method has wrong size. Set P, l_1, l_2 to be 0');
        P_o = 0;
        z_1 = 0;
        z_2 = 0;
    end
    
    covA = transpose(A)*A;
    covB = transpose(B)*B;

    %We weight A^T A by w_1 and B^T B by w_2. Note that A^T A = summation of
    %v_i v_i^T over vector v_i in group A, so w_1 A^T A can be obtained by
    %scaling each v_i to sqrt(w_1) v_i. Similar for group B.
    
    coeff_P_o = pca( [ (sqrt((1/m_A)*w_1)).*A ; (sqrt((1/m_B)*w_2)).*B], 'NumComponents', d);
    
    %coeff_P_o is now an n x d matrix
    P_o = coeff_P_o * transpose(coeff_P_o);
    z_1 = (1/m_A)*(alpha - sum(sum( covA .* P_o )));
    z_2 = (1/m_B)*(beta - sum(sum( covB .* P_o )));
end


function [Mhat] = optApprox(M, d)
    %UNTITLED3 Summary of this function goes here
    %   Detailed explanation goes here
    
    coeff = pca(M, 'NumComponents', d);
    P = coeff * transpose(coeff);
    Mhat = M*P;
end