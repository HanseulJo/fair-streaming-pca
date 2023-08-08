% Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0


%This script is an adaptation of 
%https://github.com/nick-jhlee/fair-manifold-pca/blob/master/uci/uci_fpca_analysis.m
%
%The function 'fairness_metric' defined at the bottom of this script is
%copied from https://github.com/nick-jhlee/fair-manifold-pca/blob/master/fairness_metric.m
%
%The functions 'mmd' and 'K' defined at the bottom of this script are
%copied from https://github.com/nick-jhlee/fair-manifold-pca/blob/master/mmd.m


mkdir('./fair-manifold-pca/uci/StandardPCA')
mkdir('./fair-manifold-pca/uci/FPCA00')
mkdir('./fair-manifold-pca/uci/FPCA01')
mkdir('./fair-manifold-pca/uci/MBFPCA3')
mkdir('./fair-manifold-pca/uci/MBFPCA6')
mkdir('./fair-manifold-pca/uci/FairPCA-S05')
mkdir('./fair-manifold-pca/uci/FairPCA-S085')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-offline-mean-')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-offline-all-2')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-offline-all-5')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-offline-all-10')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-offline-all-15')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-offline-all-25')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-offline-all-50')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-iterative-mean-')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-iterative-all-2')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-iterative-all-5')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-iterative-all-10')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-iterative-all-15')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-iterative-all-25')
mkdir('./fair-manifold-pca/uci/FairStreamingPCA-iterative-all-50')
methods = {'Fair PCA', 'Fair Kernel PCA', 'INLP', 'RLACE', 'PCA Samadi'};
for method_num=1:length(methods)
    method=methods{method_num};
    mkdir(sprintf('./fair-manifold-pca/uci/%s',strrep(method," ","")))
end

names = {'COMPAS', 'German', 'Adult'};
% names = {'COMPAS', 'German'};
for embedding_dim=[2,10]
    
    mmds_PCA = zeros(10, 3);
    exp_vars_PCA = zeros(10, 3);
    accs_PCA = zeros(10, 3);
    DPs_PCA = zeros(10, 3);
    accs_PCA_linear = zeros(10, 3);
    DPs_PCA_linear = zeros(10, 3);
    
    mmds_FPCA00 = zeros(10, 3);
    exp_vars_FPCA00 = zeros(10, 3);
    accs_FPCA00 = zeros(10, 3);
    DPs_FPCA00 = zeros(10, 3);
    accs_FPCA00_linear = zeros(10, 3);
    DPs_FPCA00_linear = zeros(10, 3);
    
    mmds_FPCA01 = zeros(10, 3);
    exp_vars_FPCA01 = zeros(10, 3);
    accs_FPCA01 = zeros(10, 3);
    DPs_FPCA01 = zeros(10, 3);
    accs_FPCA01_linear = zeros(10, 3);
    DPs_FPCA01_linear = zeros(10, 3);
    
    mmds_MBFPCA3 = zeros(10, 3);
    exp_vars_MBFPCA3 = zeros(10, 3);
    accs_MBFPCA3 = zeros(10, 3);
    DPs_MBFPCA3 = zeros(10, 3);
    accs_MBFPCA3_linear = zeros(10, 3);
    DPs_MBFPCA3_linear = zeros(10, 3);
    
    mmds_MBFPCA6 = zeros(10, 3);
    exp_vars_MBFPCA6 = zeros(10, 3);
    accs_MBFPCA6 = zeros(10, 3);
    DPs_MBFPCA6 = zeros(10, 3);
    accs_MBFPCA6_linear = zeros(10, 3);
    DPs_MBFPCA6_linear = zeros(10, 3);
    
    mmds_FairPCA = zeros(10, 3);
    exp_vars_FairPCA = zeros(10, 3);
    accs_FairPCA = zeros(10, 3);
    DPs_FairPCA = zeros(10, 3);
    accs_FairPCA_linear = zeros(10, 3);
    DPs_FairPCA_linear = zeros(10, 3);

    mmds_FairPCA_S05 = zeros(10, 3);
    exp_vars_FairPCA_S05 = zeros(10, 3);
    accs_FairPCA_S05 = zeros(10, 3);
    DPs_FairPCA_S05 = zeros(10, 3);
    accs_FairPCA_S05_linear = zeros(10, 3);
    DPs_FairPCA_S05_linear = zeros(10, 3);

    mmds_FairPCA_S085 = zeros(10, 3);
    exp_vars_FairPCA_S085 = zeros(10, 3);
    accs_FairPCA_S085 = zeros(10, 3);
    DPs_FairPCA_S085 = zeros(10, 3);
    accs_FairPCA_S085_linear = zeros(10, 3);
    DPs_FairPCA_S085_linear = zeros(10, 3);

    accs_FairKernelPCA = zeros(10, 3);
    DPs_FairKernelPCA = zeros(10, 3);
    accs_FairKernelPCA_linear = zeros(10, 3);
    DPs_FairKernelPCA_linear = zeros(10, 3);

    mmds_INLP = zeros(10, 3);
    exp_vars_INLP = zeros(10, 3);
    accs_INLP = zeros(10, 3);
    DPs_INLP = zeros(10, 3);
    accs_INLP_linear = zeros(10, 3);
    DPs_INLP_linear = zeros(10, 3);
    
    mmds_RLACE = zeros(10, 3);
    exp_vars_RLACE = zeros(10, 3);
    accs_RLACE = zeros(10, 3);
    DPs_RLACE = zeros(10, 3);
    accs_RLACE_linear = zeros(10, 3);
    DPs_RLACE_linear = zeros(10, 3);
    
    accs_PCA_Samadi = zeros(10, 3);
    DPs_PCA_Samadi = zeros(10, 3);
    accs_PCA_Samadi_linear = zeros(10, 3);
    DPs_PCA_Samadi_linear = zeros(10, 3);

    mmds_FairStreamingPCAofflineMean = zeros(10, 3);
    exp_vars_FairStreamingPCAofflineMean = zeros(10, 3);
    accs_FairStreamingPCAofflineMean = zeros(10, 3);
    DPs_FairStreamingPCAofflineMean = zeros(10, 3);
    accs_FairStreamingPCAofflineMean_linear = zeros(10, 3);
    DPs_FairStreamingPCAofflineMean_linear = zeros(10, 3);

    mmds_FairStreamingPCAofflineAll2 = zeros(10, 3);
    exp_vars_FairStreamingPCAofflineAll2 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll2 = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll2 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll2_linear = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll2_linear = zeros(10, 3);

    mmds_FairStreamingPCAofflineAll5 = zeros(10, 3);
    exp_vars_FairStreamingPCAofflineAll5 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll5 = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll5 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll5_linear = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll5_linear = zeros(10, 3);

    mmds_FairStreamingPCAofflineAll10 = zeros(10, 3);
    exp_vars_FairStreamingPCAofflineAll10 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll10 = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll10 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll10_linear = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll10_linear = zeros(10, 3);

    mmds_FairStreamingPCAofflineAll15 = zeros(10, 3);
    exp_vars_FairStreamingPCAofflineAll15 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll15 = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll15 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll15_linear = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll15_linear = zeros(10, 3);

    mmds_FairStreamingPCAofflineAll25 = zeros(10, 3);
    exp_vars_FairStreamingPCAofflineAll25 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll25 = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll25 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll25_linear = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll25_linear = zeros(10, 3);

    mmds_FairStreamingPCAofflineAll50 = zeros(10, 3);
    exp_vars_FairStreamingPCAofflineAll50 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll50 = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll50 = zeros(10, 3);
    accs_FairStreamingPCAofflineAll50_linear = zeros(10, 3);
    DPs_FairStreamingPCAofflineAll50_linear = zeros(10, 3);

    mmds_FairStreamingPCAiterativeMean = zeros(10, 3);
    exp_vars_FairStreamingPCAiterativeMean = zeros(10, 3);
    accs_FairStreamingPCAiterativeMean = zeros(10, 3);
    DPs_FairStreamingPCAiterativeMean = zeros(10, 3);
    accs_FairStreamingPCAiterativeMean_linear = zeros(10, 3);
    DPs_FairStreamingPCAiterativeMean_linear = zeros(10, 3);

    mmds_FairStreamingPCAiterativeAll2 = zeros(10, 3);
    exp_vars_FairStreamingPCAiterativeAll2 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll2 = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll2 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll2_linear = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll2_linear = zeros(10, 3);

    mmds_FairStreamingPCAiterativeAll5 = zeros(10, 3);
    exp_vars_FairStreamingPCAiterativeAll5 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll5 = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll5 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll5_linear = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll5_linear = zeros(10, 3);

    mmds_FairStreamingPCAiterativeAll10 = zeros(10, 3);
    exp_vars_FairStreamingPCAiterativeAll10 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll10 = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll10 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll10_linear = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll10_linear = zeros(10, 3);

    mmds_FairStreamingPCAiterativeAll15 = zeros(10, 3);
    exp_vars_FairStreamingPCAiterativeAll15 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll15 = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll15 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll15_linear = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll15_linear = zeros(10, 3);

    mmds_FairStreamingPCAiterativeAll25 = zeros(10, 3);
    exp_vars_FairStreamingPCAiterativeAll25 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll25 = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll25 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll25_linear = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll25_linear = zeros(10, 3);

    mmds_FairStreamingPCAiterativeAll50 = zeros(10, 3);
    exp_vars_FairStreamingPCAiterativeAll50 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll50 = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll50 = zeros(10, 3);
    accs_FairStreamingPCAiterativeAll50_linear = zeros(10, 3);
    DPs_FairStreamingPCAiterativeAll50_linear = zeros(10, 3);

  
    for name_num = 1:3
        for split = 1:10
            
            %% Load datas
            X_train = table2array(readtable(sprintf('fair-manifold-pca/datasets/%s/train_%d.csv', names{name_num}, split-1)));
            Y_train = X_train(:, end-1);
            Z_train = X_train(:, end);
            X_train = X_train(:, 1:end-2);
            
            X = readmatrix(sprintf('fair-manifold-pca/datasets/%s/test_%d.csv', names{name_num}, split-1));
            
            % if name_num == 2
            %     V_FPCA00 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FPCA00_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            %     V_FPCA01 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FPCA01_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % end
            if ~(name_num==3 && embedding_dim==10) 
                V_MBFPCA3 = readmatrix(sprintf('fair-manifold-pca/uci/%s/MBFPCA3_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
                V_MBFPCA6 = readmatrix(sprintf('fair-manifold-pca/uci/%s/MBFPCA6_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            end
            % V_FairPCA = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairPCA_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairPCA_S05 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairPCA_S05_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairPCA_S085 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairPCA_S085_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % embedding_FairKernelPCA_test = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairKernelPCA_embedding_TEST_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_INLP = readmatrix(sprintf('fair-manifold-pca/uci/%s/INLP_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_RLACE = readmatrix(sprintf('fair-manifold-pca/uci/%s/RLACE_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % embedding_PCASamadi_test = readmatrix(sprintf('fair-manifold-pca/uci/%s/PCA_Samadi_embedding_TEST_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAofflineMean = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-offline-mean-_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAofflineAll2 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-offline-all-2_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAofflineAll5 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-offline-all-5_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAofflineAll10 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-offline-all-10_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAofflineAll15 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-offline-all-15_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAofflineAll25 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-offline-all-25_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAofflineAll50 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-offline-all-50_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAiterativeMean = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-iterative-mean-_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAiterativeAll2 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-iterative-all-2_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAiterativeAll5 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-iterative-all-5_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAiterativeAll10 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-iterative-all-10_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAiterativeAll15 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-iterative-all-15_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAiterativeAll25 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-iterative-all-25_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));
            % V_FairStreamingPCAiterativeAll50 = readmatrix(sprintf('fair-manifold-pca/uci/%s/FairStreamingPCA-iterative-all-50_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim));

            Y = X(:, end-1);
            Z = X(:, end);
            n1 = sum(Z);
            n2 = sum(Z == 0);
            X = X(:, 1:end-2);
            A = cov(X);


            %% Obtain PCA and sigma
            V_pca = pca(X_train);
            V_pca = V_pca(:, 1:embedding_dim);
            % writematrix(V_pca, sprintf('fair-manifold-pca/uci/%s/PCA_V_%d_dim%d.csv', names{name_num}, split-1, embedding_dim))
    
            % Obtain sigma
            sigma = sqrt(median(pdist(X_train*V_pca, 'squaredeuclidean'))/2);

            %% Store PCA results
            % mmds_PCA(split, name_num) = mmd(X(Z==1,:)*V_pca, X(Z==0,:)*V_pca, sigma);
            % exp_vars_PCA(split, name_num) = 100 * trace(V_pca'*A*V_pca)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_pca, Y, Z);
            % accs_PCA(split, name_num) = acc;
            % DPs_PCA(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_pca, Y, Z);
            % accs_PCA_linear(split, name_num) = acc;
            % DPs_PCA_linear(split, name_num) = DP;
            
            %% Store FPCA results
            % if name_num == 2
            %     mmds_FPCA00(split, name_num) = mmd(X(Z==1,:)*V_FPCA00, X(Z==0,:)*V_FPCA00, sigma);
            %     exp_vars_FPCA00(split, name_num) = 100 * trace(V_FPCA00'*A*V_FPCA00)/trace(A);
            %     mmds_FPCA01(split, name_num) = mmd(X(Z==1,:)*V_FPCA01, X(Z==0,:)*V_FPCA01, sigma);
            %     exp_vars_FPCA01(split, name_num) = 100 * trace(V_FPCA01'*A*V_FPCA01)/trace(A);
            % 
            %     % fairness metrics
            %     [acc, DP, ~, ~] = fairness_metric(X*V_FPCA00, Y, Z);
            %     accs_FPCA00(split, name_num) = acc;
            %     DPs_FPCA00(split, name_num) = DP;
            % 
            %     [acc, DP, ~, ~] = fairness_metric(X*V_FPCA01, Y, Z);
            %     accs_FPCA01(split, name_num) = acc;
            %     DPs_FPCA01(split, name_num) = DP;
            % 
            %     % fairness metrics linear
            %     [acc, DP, ~, ~] = fairness_metric_linear(X*V_FPCA00, Y, Z);
            %     accs_FPCA00_linear(split, name_num) = acc;
            %     DPs_FPCA00_linear(split, name_num) = DP;
            % 
            %     [acc, DP, ~, ~] = fairness_metric_linear(X*V_FPCA01, Y, Z);
            %     accs_FPCA01_linear(split, name_num) = acc;
            %     DPs_FPCA01_linear(split, name_num) = DP;
            % end
            %% Store MBFPCA results
            if ~(name_num==3 && embedding_dim==10)
                disp('MBFPCA');
                mmds_MBFPCA3(split, name_num) = mmd(X(Z==1,:)*V_MBFPCA3, X(Z==0,:)*V_MBFPCA3, sigma);
                exp_vars_MBFPCA3(split, name_num) = 100 * trace(V_MBFPCA3'*A*V_MBFPCA3)/trace(A);
                mmds_MBFPCA6(split, name_num) = mmd(X(Z==1,:)*V_MBFPCA6, X(Z==0,:)*V_MBFPCA6, sigma);
                exp_vars_MBFPCA6(split, name_num) = 100 * trace(V_MBFPCA6'*A*V_MBFPCA6)/trace(A);

                % fairness metrics
                [acc, DP, ~, ~] = fairness_metric(X*V_MBFPCA3, Y, Z);
                accs_MBFPCA3(split, name_num) = acc;
                DPs_MBFPCA3(split, name_num) = DP;

                [acc, DP, ~, ~] = fairness_metric(X*V_MBFPCA6, Y, Z);
                accs_MBFPCA6(split, name_num) = acc;
                DPs_MBFPCA6(split, name_num) = DP;

                % fairness metrics linear
                [acc, DP, ~, ~] = fairness_metric_linear(X*V_MBFPCA3, Y, Z);
                accs_MBFPCA3_linear(split, name_num) = acc;
                DPs_MBFPCA3_linear(split, name_num) = DP;

                [acc, DP, ~, ~] = fairness_metric_linear(X*V_MBFPCA6, Y, Z);
                accs_MBFPCA6_linear(split, name_num) = acc;
                DPs_MBFPCA6_linear(split, name_num) = DP;
                disp(mmds_MBFPCA6);
            end
            
            % %% Store FairPCA results
            % mmds_FairPCA(split, name_num) = mmd(X(Z==1,:)*V_FairPCA, X(Z==0,:)*V_FairPCA, sigma);
            % exp_vars_FairPCA(split, name_num) = 100 * trace(V_FairPCA'*A*V_FairPCA)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairPCA, Y, Z);
            % accs_FairPCA(split, name_num) = acc;
            % DPs_FairPCA(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairPCA, Y, Z);
            % accs_FairPCA_linear(split, name_num) = acc;
            % DPs_FairPCA_linear(split, name_num) = DP;
            % 
            % %% Store FairPCA-S05 results
            % mmds_FairPCA_S05(split, name_num) = mmd(X(Z==1,:)*V_FairPCA_S05, X(Z==0,:)*V_FairPCA_S05, sigma);
            % exp_vars_FairPCA_S05(split, name_num) = 100 * trace(V_FairPCA_S05'*A*V_FairPCA_S05)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairPCA_S05, Y, Z);
            % accs_FairPCA_S05(split, name_num) = acc;
            % DPs_FairPCA_S05(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairPCA_S05, Y, Z);
            % accs_FairPCA_S05_linear(split, name_num) = acc;
            % DPs_FairPCA_S05_linear(split, name_num) = DP;
            % 
            % %% Store FairPCA-S085 results
            % mmds_FairPCA_S085(split, name_num) = mmd(X(Z==1,:)*V_FairPCA_S085, X(Z==0,:)*V_FairPCA_S085, sigma);
            % exp_vars_FairPCA_S085(split, name_num) = 100 * trace(V_FairPCA_S085'*A*V_FairPCA_S085)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairPCA_S085, Y, Z);
            % accs_FairPCA_S085(split, name_num) = acc;
            % DPs_FairPCA_S085(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairPCA_S085, Y, Z);
            % accs_FairPCA_S085_linear(split, name_num) = acc;
            % DPs_FairPCA_S085_linear(split, name_num) = DP;
            % 
            % %% Store FairKernelPCA results 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(embedding_FairKernelPCA_test, Y, Z);
            % accs_FairKernelPCA(split, name_num) = acc;
            % DPs_FairKernelPCA(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(embedding_FairKernelPCA_test, Y, Z);
            % accs_FairKernelPCA_linear(split, name_num) = acc;
            % DPs_FairKernelPCA_linear(split, name_num) = DP;
            % 
            % %% Store INLP results
            % mmds_INLP(split, name_num) = mmd(X(Z==1,:)*V_INLP, X(Z==0,:)*V_INLP, sigma);
            % exp_vars_INLP(split, name_num) = 100 * trace(V_INLP'*A*V_INLP)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_INLP, Y, Z);
            % accs_INLP(split, name_num) = acc;
            % DPs_INLP(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_INLP, Y, Z);
            % accs_INLP_linear(split, name_num) = acc;
            % DPs_INLP_linear(split, name_num) = DP;
            % 
            % %% Store RLACE results
            % mmds_RLACE(split, name_num) = mmd(X(Z==1,:)*V_RLACE, X(Z==0,:)*V_RLACE, sigma);
            % exp_vars_RLACE(split, name_num) = 100 * trace(V_RLACE'*A*V_RLACE)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_RLACE, Y, Z);
            % accs_RLACE(split, name_num) = acc;
            % DPs_RLACE(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_RLACE, Y, Z);
            % accs_RLACE_linear(split, name_num) = acc;
            % DPs_RLACE_linear(split, name_num) = DP;
            % 
            % %% Store PCASamadi results 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(embedding_PCASamadi_test, Y, Z);
            % accs_PCA_Samadi(split, name_num) = acc;
            % DPs_PCA_Samadi(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(embedding_PCASamadi_test, Y, Z);
            % accs_PCA_Samadi_linear(split, name_num) = acc;
            % DPs_PCA_Samadi_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (offline, mean) results
            % disp('FairStreamingPCA (offline, mean)');
            % mmds_FairStreamingPCAofflineMean(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAofflineMean, X(Z==0,:)*V_FairStreamingPCAofflineMean, sigma);
            % exp_vars_FairStreamingPCAofflineMean(split, name_num) = 100 * trace(V_FairStreamingPCAofflineMean'*A*V_FairStreamingPCAofflineMean)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAofflineMean, Y, Z);
            % accs_FairStreamingPCAofflineMean(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineMean(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAofflineMean, Y, Z);
            % accs_FairStreamingPCAofflineMean_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineMean_linear(split, name_num) = DP;
            % disp(mmds_FairStreamingPCAofflineMean)

            % %% Store FairStreamingPCA (offline, all, 2) results
            % mmds_FairStreamingPCAofflineAll2(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAofflineAll2, X(Z==0,:)*V_FairStreamingPCAofflineAll2, sigma);
            % exp_vars_FairStreamingPCAofflineAll2(split, name_num) = 100 * trace(V_FairStreamingPCAofflineAll2'*A*V_FairStreamingPCAofflineAll2)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAofflineAll2, Y, Z);
            % accs_FairStreamingPCAofflineAll2(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll2(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAofflineAll2, Y, Z);
            % accs_FairStreamingPCAofflineAll2_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll2_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (offline, all, 5) results
            % mmds_FairStreamingPCAofflineAll5(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAofflineAll5, X(Z==0,:)*V_FairStreamingPCAofflineAll5, sigma);
            % exp_vars_FairStreamingPCAofflineAll5(split, name_num) = 100 * trace(V_FairStreamingPCAofflineAll5'*A*V_FairStreamingPCAofflineAll5)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAofflineAll5, Y, Z);
            % accs_FairStreamingPCAofflineAll5(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll5(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAofflineAll5, Y, Z);
            % accs_FairStreamingPCAofflineAll5_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll5_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (offline, all, 10) results
            % mmds_FairStreamingPCAofflineAll10(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAofflineAll10, X(Z==0,:)*V_FairStreamingPCAofflineAll10, sigma);
            % exp_vars_FairStreamingPCAofflineAll10(split, name_num) = 100 * trace(V_FairStreamingPCAofflineAll10'*A*V_FairStreamingPCAofflineAll10)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAofflineAll10, Y, Z);
            % accs_FairStreamingPCAofflineAll10(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll10(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAofflineAll10, Y, Z);
            % accs_FairStreamingPCAofflineAll10_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll10_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (offline, all, 15) results
            % mmds_FairStreamingPCAofflineAll15(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAofflineAll15, X(Z==0,:)*V_FairStreamingPCAofflineAll15, sigma);
            % exp_vars_FairStreamingPCAofflineAll15(split, name_num) = 100 * trace(V_FairStreamingPCAofflineAll15'*A*V_FairStreamingPCAofflineAll15)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAofflineAll15, Y, Z);
            % accs_FairStreamingPCAofflineAll15(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll15(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAofflineAll15, Y, Z);
            % accs_FairStreamingPCAofflineAll15_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll15_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (offline, all, 25) results
            % mmds_FairStreamingPCAofflineAll25(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAofflineAll25, X(Z==0,:)*V_FairStreamingPCAofflineAll2, sigma);
            % exp_vars_FairStreamingPCAofflineAll25(split, name_num) = 100 * trace(V_FairStreamingPCAofflineAll25'*A*V_FairStreamingPCAofflineAll25)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAofflineAll25, Y, Z);
            % accs_FairStreamingPCAofflineAll25(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll25(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAofflineAll25, Y, Z);
            % accs_FairStreamingPCAofflineAll25_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll25_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (offline, all, 50) results
            % mmds_FairStreamingPCAofflineAll50(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAofflineAll50, X(Z==0,:)*V_FairStreamingPCAofflineAll50, sigma);
            % exp_vars_FairStreamingPCAofflineAll50(split, name_num) = 100 * trace(V_FairStreamingPCAofflineAll50'*A*V_FairStreamingPCAofflineAll50)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAofflineAll50, Y, Z);
            % accs_FairStreamingPCAofflineAll50(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll50(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAofflineAll50, Y, Z);
            % accs_FairStreamingPCAofflineAll50_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAofflineAll50_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (iterative, mean) results
            % mmds_FairStreamingPCAiterativeMean(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAiterativeMean, X(Z==0,:)*V_FairStreamingPCAiterativeMean, sigma);
            % exp_vars_FairStreamingPCAiterativeMean(split, name_num) = 100 * trace(V_FairStreamingPCAiterativeMean'*A*V_FairStreamingPCAiterativeMean)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAiterativeMean, Y, Z);
            % accs_FairStreamingPCAiterativeMean(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeMean(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAiterativeMean, Y, Z);
            % accs_FairStreamingPCAiterativeMean_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeMean_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (iterative, all, 2) results
            % mmds_FairStreamingPCAiterativeAll2(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAiterativeAll2, X(Z==0,:)*V_FairStreamingPCAiterativeAll2, sigma);
            % exp_vars_FairStreamingPCAiterativeAll2(split, name_num) = 100 * trace(V_FairStreamingPCAiterativeAll2'*A*V_FairStreamingPCAiterativeAll2)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAiterativeAll2, Y, Z);
            % accs_FairStreamingPCAiterativeAll2(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll2(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAiterativeAll2, Y, Z);
            % accs_FairStreamingPCAiterativeAll2_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll2_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (iterative, all, 5) results
            % mmds_FairStreamingPCAiterativeAll5(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAiterativeAll5, X(Z==0,:)*V_FairStreamingPCAiterativeAll5, sigma);
            % exp_vars_FairStreamingPCAiterativeAll5(split, name_num) = 100 * trace(V_FairStreamingPCAiterativeAll5'*A*V_FairStreamingPCAiterativeAll5)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAiterativeAll5, Y, Z);
            % accs_FairStreamingPCAiterativeAll5(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll5(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAiterativeAll5, Y, Z);
            % accs_FairStreamingPCAiterativeAll5_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll5_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (iterative, all, 10) results
            % mmds_FairStreamingPCAiterativeAll10(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAiterativeAll10, X(Z==0,:)*V_FairStreamingPCAiterativeAll10, sigma);
            % exp_vars_FairStreamingPCAiterativeAll10(split, name_num) = 100 * trace(V_FairStreamingPCAiterativeAll10'*A*V_FairStreamingPCAiterativeAll10)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAiterativeAll10, Y, Z);
            % accs_FairStreamingPCAiterativeAll10(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll10(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAiterativeAll10, Y, Z);
            % accs_FairStreamingPCAiterativeAll10_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll10_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (iterative, all, 15) results
            % mmds_FairStreamingPCAiterativeAll15(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAiterativeAll15, X(Z==0,:)*V_FairStreamingPCAiterativeAll15, sigma);
            % exp_vars_FairStreamingPCAiterativeAll15(split, name_num) = 100 * trace(V_FairStreamingPCAiterativeAll15'*A*V_FairStreamingPCAiterativeAll15)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAiterativeAll15, Y, Z);
            % accs_FairStreamingPCAiterativeAll15(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll15(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAiterativeAll15, Y, Z);
            % accs_FairStreamingPCAiterativeAll15_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll15_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (iterative, all, 25) results
            % mmds_FairStreamingPCAiterativeAll25(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAiterativeAll25, X(Z==0,:)*V_FairStreamingPCAiterativeAll2, sigma);
            % exp_vars_FairStreamingPCAiterativeAll25(split, name_num) = 100 * trace(V_FairStreamingPCAiterativeAll25'*A*V_FairStreamingPCAiterativeAll25)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAiterativeAll25, Y, Z);
            % accs_FairStreamingPCAiterativeAll25(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll25(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAiterativeAll25, Y, Z);
            % accs_FairStreamingPCAiterativeAll25_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll25_linear(split, name_num) = DP;
            % 
            % %% Store FairStreamingPCA (iterative, all, 50) results
            % mmds_FairStreamingPCAiterativeAll50(split, name_num) = mmd(X(Z==1,:)*V_FairStreamingPCAiterativeAll50, X(Z==0,:)*V_FairStreamingPCAiterativeAll50, sigma);
            % exp_vars_FairStreamingPCAiterativeAll50(split, name_num) = 100 * trace(V_FairStreamingPCAiterativeAll50'*A*V_FairStreamingPCAiterativeAll50)/trace(A);
            % 
            % % fairness metrics
            % [acc, DP, ~, ~] = fairness_metric(X*V_FairStreamingPCAiterativeAll50, Y, Z);
            % accs_FairStreamingPCAiterativeAll50(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll50(split, name_num) = DP;
            % 
            % % fairness metrics linear
            % [acc, DP, ~, ~] = fairness_metric_linear(X*V_FairStreamingPCAiterativeAll50, Y, Z);
            % accs_FairStreamingPCAiterativeAll50_linear(split, name_num) = acc;
            % DPs_FairStreamingPCAiterativeAll50_linear(split, name_num) = DP;
        end
    end

    % writematrix([mean(mmds_PCA,1); std(mmds_PCA,1)] , sprintf('fair-manifold-pca/uci/StandardPCA/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_PCA,1); std(exp_vars_PCA,1)] , sprintf('fair-manifold-pca/uci/StandardPCA/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_PCA,1); std(accs_PCA,1)], sprintf('fair-manifold-pca/uci/StandardPCA/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_PCA,1); std(DPs_PCA,1)], sprintf('fair-manifold-pca/uci/StandardPCA/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_PCA_linear,1); std(accs_PCA_linear,1)], sprintf('fair-manifold-pca/uci/StandardPCA/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_PCA_linear,1); std(DPs_PCA_linear,1)], sprintf('fair-manifold-pca/uci/StandardPCA/DPs_linear_dim%d.csv', embedding_dim))
    
    % writematrix([mean(mmds_FPCA00,1); std(mmds_FPCA00,1)] , sprintf('fair-manifold-pca/uci/FPCA00/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FPCA00,1); std(exp_vars_FPCA00,1)] , sprintf('fair-manifold-pca/uci/FPCA00/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FPCA00,1); std(accs_FPCA00,1)], sprintf('fair-manifold-pca/uci/FPCA00/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FPCA00,1); std(DPs_FPCA00,1)], sprintf('fair-manifold-pca/uci/FPCA00/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FPCA00_linear,1); std(accs_FPCA00_linear,1)], sprintf('fair-manifold-pca/uci/FPCA00/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FPCA00_linear,1); std(DPs_FPCA00_linear,1)], sprintf('fair-manifold-pca/uci/FPCA00/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FPCA01,1); std(mmds_FPCA01,1)] , sprintf('fair-manifold-pca/uci/FPCA01/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FPCA01,1); std(exp_vars_FPCA01,1)] , sprintf('fair-manifold-pca/uci/FPCA01/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FPCA01,1); std(accs_FPCA01,1)], sprintf('fair-manifold-pca/uci/FPCA01/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FPCA01,1); std(DPs_FPCA01,1)], sprintf('fair-manifold-pca/uci/FPCA01/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FPCA01_linear,1); std(accs_FPCA01_linear,1)], sprintf('fair-manifold-pca/uci/FPCA01/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FPCA01_linear,1); std(DPs_FPCA01_linear,1)], sprintf('fair-manifold-pca/uci/FPCA01/DPs_linear_dim%d.csv', embedding_dim))

    writematrix([mean(mmds_MBFPCA3,1); std(mmds_MBFPCA3,1)] , sprintf('fair-manifold-pca/uci/MBFPCA3/mmds_test_dim%d.csv', embedding_dim))
    writematrix([mean(exp_vars_MBFPCA3,1); std(exp_vars_MBFPCA3,1)] , sprintf('fair-manifold-pca/uci/MBFPCA3/exp_vars_test_dim%d.csv', embedding_dim))
    writematrix([mean(accs_MBFPCA3,1); std(accs_MBFPCA3,1)], sprintf('fair-manifold-pca/uci/MBFPCA3/accs_dim%d.csv', embedding_dim))
    writematrix([mean(DPs_MBFPCA3,1); std(DPs_MBFPCA3,1)], sprintf('fair-manifold-pca/uci/MBFPCA3/DPs_dim%d.csv', embedding_dim))
    writematrix([mean(accs_MBFPCA3_linear,1); std(accs_MBFPCA3_linear,1)], sprintf('fair-manifold-pca/uci/MBFPCA3/accs_linear_dim%d.csv', embedding_dim))
    writematrix([mean(DPs_MBFPCA3_linear,1); std(DPs_MBFPCA3_linear,1)], sprintf('fair-manifold-pca/uci/MBFPCA3/DPs_linear_dim%d.csv', embedding_dim))

    writematrix([mean(mmds_MBFPCA6,1); std(mmds_MBFPCA6,1)] , sprintf('fair-manifold-pca/uci/MBFPCA6/mmds_test_dim%d.csv', embedding_dim))
    writematrix([mean(exp_vars_MBFPCA6,1); std(exp_vars_MBFPCA6,1)] , sprintf('fair-manifold-pca/uci/MBFPCA6/exp_vars_test_dim%d.csv', embedding_dim))
    writematrix([mean(accs_MBFPCA6,1); std(accs_MBFPCA6,1)], sprintf('fair-manifold-pca/uci/MBFPCA6/accs_dim%d.csv', embedding_dim))
    writematrix([mean(DPs_MBFPCA6,1); std(DPs_MBFPCA6,1)], sprintf('fair-manifold-pca/uci/MBFPCA6/DPs_dim%d.csv', embedding_dim))
    writematrix([mean(accs_MBFPCA6_linear,1); std(accs_MBFPCA6_linear,1)], sprintf('fair-manifold-pca/uci/MBFPCA6/accs_linear_dim%d.csv', embedding_dim))
    writematrix([mean(DPs_MBFPCA6_linear,1); std(DPs_MBFPCA6_linear,1)], sprintf('fair-manifold-pca/uci/MBFPCA6/DPs_linear_dim%d.csv', embedding_dim))
     
    % writematrix([mean(mmds_FairPCA,1); std(mmds_FairPCA,1)] , sprintf('fair-manifold-pca/uci/FairPCA/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairPCA,1); std(exp_vars_FairPCA,1)] , sprintf('fair-manifold-pca/uci/FairPCA/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairPCA,1); std(accs_FairPCA,1)], sprintf('fair-manifold-pca/uci/FairPCA/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairPCA,1); std(DPs_FairPCA,1)], sprintf('fair-manifold-pca/uci/FairPCA/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairPCA_linear,1); std(accs_FairPCA_linear,1)], sprintf('fair-manifold-pca/uci/FairPCA/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairPCA_linear,1); std(DPs_FairPCA_linear,1)], sprintf('fair-manifold-pca/uci/FairPCA/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairPCA_S05,1); std(mmds_FairPCA_S05,1)] , sprintf('fair-manifold-pca/uci/FairPCA-S05/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairPCA_S05,1); std(exp_vars_FairPCA_S05,1)] , sprintf('fair-manifold-pca/uci/FairPCA-S05/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairPCA_S05,1); std(accs_FairPCA_S05,1)], sprintf('fair-manifold-pca/uci/FairPCA-S05/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairPCA_S05,1); std(DPs_FairPCA_S05,1)], sprintf('fair-manifold-pca/uci/FairPCA-S05/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairPCA_S05_linear,1); std(accs_FairPCA_S05_linear,1)], sprintf('fair-manifold-pca/uci/FairPCA-S05/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairPCA_S05_linear,1); std(DPs_FairPCA_S05_linear,1)], sprintf('fair-manifold-pca/uci/FairPCA-S05/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairPCA_S085,1); std(mmds_FairPCA_S085,1)] , sprintf('fair-manifold-pca/uci/FairPCA-S085/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairPCA_S085,1); std(exp_vars_FairPCA_S085,1)] , sprintf('fair-manifold-pca/uci/FairPCA-S085/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairPCA_S085,1); std(accs_FairPCA_S085,1)], sprintf('fair-manifold-pca/uci/FairPCA-S085/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairPCA_S085,1); std(DPs_FairPCA_S085,1)], sprintf('fair-manifold-pca/uci/FairPCA-S085/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairPCA_S085_linear,1); std(accs_FairPCA_S085_linear,1)], sprintf('fair-manifold-pca/uci/FairPCA-S085/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairPCA_S085_linear,1); std(DPs_FairPCA_S085_linear,1)], sprintf('fair-manifold-pca/uci/FairPCA-S085/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(accs_FairKernelPCA,1); std(accs_FairKernelPCA,1)], sprintf('fair-manifold-pca/uci/FairKernelPCA/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairKernelPCA,1); std(DPs_FairKernelPCA,1)], sprintf('fair-manifold-pca/uci/FairKernelPCA/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairKernelPCA_linear,1); std(accs_FairKernelPCA_linear,1)], sprintf('fair-manifold-pca/uci/FairKernelPCA/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairKernelPCA_linear,1); std(DPs_FairKernelPCA_linear,1)], sprintf('fair-manifold-pca/uci/FairKernelPCA/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_INLP,1); std(mmds_INLP,1)] , sprintf('fair-manifold-pca/uci/INLP/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_INLP,1); std(exp_vars_INLP,1)] , sprintf('fair-manifold-pca/uci/INLP/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_INLP,1); std(accs_INLP,1)], sprintf('fair-manifold-pca/uci/INLP/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_INLP,1); std(DPs_INLP,1)], sprintf('fair-manifold-pca/uci/INLP/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_INLP_linear,1); std(accs_INLP_linear,1)], sprintf('fair-manifold-pca/uci/INLP/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_INLP_linear,1); std(DPs_INLP_linear,1)], sprintf('fair-manifold-pca/uci/INLP/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_RLACE,1); std(mmds_RLACE,1)] , sprintf('fair-manifold-pca/uci/RLACE/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_RLACE,1); std(exp_vars_RLACE,1)] , sprintf('fair-manifold-pca/uci/RLACE/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_RLACE,1); std(accs_RLACE,1)], sprintf('fair-manifold-pca/uci/RLACE/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_RLACE,1); std(DPs_RLACE,1)], sprintf('fair-manifold-pca/uci/RLACE/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_RLACE_linear,1); std(accs_RLACE_linear,1)], sprintf('fair-manifold-pca/uci/RLACE/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_RLACE_linear,1); std(DPs_RLACE_linear,1)], sprintf('fair-manifold-pca/uci/RLACE/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(accs_PCA_Samadi,1); std(accs_PCA_Samadi,1)], sprintf('fair-manifold-pca/uci/PCASamadi/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_PCA_Samadi,1); std(DPs_PCA_Samadi,1)], sprintf('fair-manifold-pca/uci/PCASamadi/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_PCA_Samadi_linear,1); std(accs_PCA_Samadi_linear,1)], sprintf('fair-manifold-pca/uci/PCASamadi/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_PCA_Samadi_linear,1); std(DPs_PCA_Samadi_linear,1)], sprintf('fair-manifold-pca/uci/PCASamadi/DPs_linear_dim%d.csv', embedding_dim))

    % writematrix([mean(mmds_FairStreamingPCAofflineMean,1); std(mmds_FairStreamingPCAofflineMean,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-mean-/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAofflineMean,1); std(exp_vars_FairStreamingPCAofflineMean,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-mean-/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineMean,1); std(accs_FairStreamingPCAofflineMean,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-mean-/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineMean,1); std(DPs_FairStreamingPCAofflineMean,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-mean-/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineMean_linear,1); std(accs_FairStreamingPCAofflineMean_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-mean-/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineMean_linear,1); std(DPs_FairStreamingPCAofflineMean_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-mean-/DPs_linear_dim%d.csv', embedding_dim))

    % writematrix([mean(mmds_FairStreamingPCAofflineAll2,1); std(mmds_FairStreamingPCAofflineAll2,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-2/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAofflineAll2,1); std(exp_vars_FairStreamingPCAofflineAll2,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-2/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll2,1); std(accs_FairStreamingPCAofflineAll2,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-2/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll2,1); std(DPs_FairStreamingPCAofflineAll2,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-2/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll2_linear,1); std(accs_FairStreamingPCAofflineAll2_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-2/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll2_linear,1); std(DPs_FairStreamingPCAofflineAll2_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-2/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAofflineAll5,1); std(mmds_FairStreamingPCAofflineAll5,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-5/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAofflineAll5,1); std(exp_vars_FairStreamingPCAofflineAll5,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-5/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll5,1); std(accs_FairStreamingPCAofflineAll5,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-5/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll5,1); std(DPs_FairStreamingPCAofflineAll5,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-5/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll5_linear,1); std(accs_FairStreamingPCAofflineAll5_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-5/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll5_linear,1); std(DPs_FairStreamingPCAofflineAll5_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-5/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAofflineAll10,1); std(mmds_FairStreamingPCAofflineAll10,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-10/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAofflineAll10,1); std(exp_vars_FairStreamingPCAofflineAll10,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-10/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll10,1); std(accs_FairStreamingPCAofflineAll10,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-10/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll10,1); std(DPs_FairStreamingPCAofflineAll10,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-10/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll10_linear,1); std(accs_FairStreamingPCAofflineAll10_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-10/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll10_linear,1); std(DPs_FairStreamingPCAofflineAll10_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-10/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAofflineAll15,1); std(mmds_FairStreamingPCAofflineAll15,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-15/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAofflineAll15,1); std(exp_vars_FairStreamingPCAofflineAll15,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-15/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll15,1); std(accs_FairStreamingPCAofflineAll15,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-15/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll15,1); std(DPs_FairStreamingPCAofflineAll15,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-15/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll15_linear,1); std(accs_FairStreamingPCAofflineAll15_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-15/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll15_linear,1); std(DPs_FairStreamingPCAofflineAll15_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-15/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAofflineAll25,1); std(mmds_FairStreamingPCAofflineAll25,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-25/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAofflineAll25,1); std(exp_vars_FairStreamingPCAofflineAll25,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-25/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll25,1); std(accs_FairStreamingPCAofflineAll25,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-25/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll25,1); std(DPs_FairStreamingPCAofflineAll25,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-25/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll25_linear,1); std(accs_FairStreamingPCAofflineAll25_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-25/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll25_linear,1); std(DPs_FairStreamingPCAofflineAll25_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-25/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAofflineAll50,1); std(mmds_FairStreamingPCAofflineAll50,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-50/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAofflineAll50,1); std(exp_vars_FairStreamingPCAofflineAll50,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-50/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll50,1); std(accs_FairStreamingPCAofflineAll50,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-50/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll50,1); std(DPs_FairStreamingPCAofflineAll50,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-50/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAofflineAll50_linear,1); std(accs_FairStreamingPCAofflineAll50_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-50/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAofflineAll50_linear,1); std(DPs_FairStreamingPCAofflineAll50_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-offline-all-50/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAiterativeMean,1); std(mmds_FairStreamingPCAiterativeMean,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-mean-/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAiterativeMean,1); std(exp_vars_FairStreamingPCAiterativeMean,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-mean-/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeMean,1); std(accs_FairStreamingPCAiterativeMean,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-mean-/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeMean,1); std(DPs_FairStreamingPCAiterativeMean,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-mean-/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeMean_linear,1); std(accs_FairStreamingPCAiterativeMean_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-mean-/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeMean_linear,1); std(DPs_FairStreamingPCAiterativeMean_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-mean-/DPs_linear_dim%d.csv', embedding_dim))

    % writematrix([mean(mmds_FairStreamingPCAiterativeAll2,1); std(mmds_FairStreamingPCAiterativeAll2,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-2/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAiterativeAll2,1); std(exp_vars_FairStreamingPCAiterativeAll2,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-2/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll2,1); std(accs_FairStreamingPCAiterativeAll2,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-2/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll2,1); std(DPs_FairStreamingPCAiterativeAll2,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-2/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll2_linear,1); std(accs_FairStreamingPCAiterativeAll2_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-2/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll2_linear,1); std(DPs_FairStreamingPCAiterativeAll2_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-2/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAiterativeAll5,1); std(mmds_FairStreamingPCAiterativeAll5,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-5/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAiterativeAll5,1); std(exp_vars_FairStreamingPCAiterativeAll5,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-5/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll5,1); std(accs_FairStreamingPCAiterativeAll5,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-5/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll5,1); std(DPs_FairStreamingPCAiterativeAll5,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-5/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll5_linear,1); std(accs_FairStreamingPCAiterativeAll5_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-5/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll5_linear,1); std(DPs_FairStreamingPCAiterativeAll5_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-5/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAiterativeAll10,1); std(mmds_FairStreamingPCAiterativeAll10,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-10/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAiterativeAll10,1); std(exp_vars_FairStreamingPCAiterativeAll10,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-10/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll10,1); std(accs_FairStreamingPCAiterativeAll10,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-10/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll10,1); std(DPs_FairStreamingPCAiterativeAll10,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-10/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll10_linear,1); std(accs_FairStreamingPCAiterativeAll10_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-10/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll10_linear,1); std(DPs_FairStreamingPCAiterativeAll10_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-10/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAiterativeAll15,1); std(mmds_FairStreamingPCAiterativeAll15,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-15/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAiterativeAll15,1); std(exp_vars_FairStreamingPCAiterativeAll15,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-15/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll15,1); std(accs_FairStreamingPCAiterativeAll15,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-15/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll15,1); std(DPs_FairStreamingPCAiterativeAll15,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-15/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll15_linear,1); std(accs_FairStreamingPCAiterativeAll15_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-15/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll15_linear,1); std(DPs_FairStreamingPCAiterativeAll15_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-15/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAiterativeAll25,1); std(mmds_FairStreamingPCAiterativeAll25,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-25/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAiterativeAll25,1); std(exp_vars_FairStreamingPCAiterativeAll25,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-25/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll25,1); std(accs_FairStreamingPCAiterativeAll25,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-25/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll25,1); std(DPs_FairStreamingPCAiterativeAll25,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-25/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll25_linear,1); std(accs_FairStreamingPCAiterativeAll25_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-25/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll25_linear,1); std(DPs_FairStreamingPCAiterativeAll25_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-25/DPs_linear_dim%d.csv', embedding_dim))
    % 
    % writematrix([mean(mmds_FairStreamingPCAiterativeAll50,1); std(mmds_FairStreamingPCAiterativeAll50,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-50/mmds_test_dim%d.csv', embedding_dim))
    % writematrix([mean(exp_vars_FairStreamingPCAiterativeAll50,1); std(exp_vars_FairStreamingPCAiterativeAll50,1)] , sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-50/exp_vars_test_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll50,1); std(accs_FairStreamingPCAiterativeAll50,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-50/accs_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll50,1); std(DPs_FairStreamingPCAiterativeAll50,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-50/DPs_dim%d.csv', embedding_dim))
    % writematrix([mean(accs_FairStreamingPCAiterativeAll50_linear,1); std(accs_FairStreamingPCAiterativeAll50_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-50/accs_linear_dim%d.csv', embedding_dim))
    % writematrix([mean(DPs_FairStreamingPCAiterativeAll50_linear,1); std(DPs_FairStreamingPCAiterativeAll50_linear,1)], sprintf('fair-manifold-pca/uci/FairStreamingPCA-iterative-all-50/DPs_linear_dim%d.csv', embedding_dim))

end





function [acc, DP, EOP, EOD] = fairness_metric(X_test, Y_test, Z_test)
    % Train RBF SVM
    SVMModel = fitcsvm(X_test, Y_test, 'KernelFunction', 'RBF');
    
    % Obtain labels of test data
    [label, ~] = predict(SVMModel, X_test);
    
    %% downstream accuracy
    acc = (1/size(X_test, 1))*sum(label == Y_test);
    
    %% fairness metrics
    % Split datas
    X1_ = X_test(Z_test == 1, :);
    X2_ = X_test(Z_test == 0, :);
    
    % DP
    DP = abs((1/size(X1_, 1))*sum(label(Z_test == 1)) - (1/size(X2_, 1))*sum(label(Z_test == 0)));
    
    % EOP
    EOP = abs((1/size(X1_, 1))*sum(label((Y_test == 1) & (Z_test == 1))) -...
        (1/size(X2_, 1))*sum(label((Y_test == 1) & (Z_test == 0))));
    
    % EOD
    EOD = max(EOP, ...
        abs((1/size(X1_, 1))*sum(label(Z_test == 1)) -...
        (1/size(X2_, 1))*sum(label(Z_test == 0))));
end

function [acc, DP, EOP, EOD] = fairness_metric_linear(X_test, Y_test, Z_test)
    % Train linear SVM
    SVMModel = fitcsvm(X_test, Y_test, 'KernelFunction', 'linear');
    [label, ~] = predict(SVMModel, X_test);
    
    %% downstream accuracy
    acc = (1/size(X_test, 1))*sum(label == Y_test);
    
    %% fairness metrics
    % Split datas
    X1_ = X_test(Z_test == 1, :);
    X2_ = X_test(Z_test == 0, :);
    
    % DP
    DP = abs((1/size(X1_, 1))*sum(label(Z_test == 1)) - (1/size(X2_, 1))*sum(label(Z_test == 0)));
    
    % EOP
    EOP = abs((1/size(X1_, 1))*sum(label((Y_test == 1) & (Z_test == 1))) -...
        (1/size(X2_, 1))*sum(label((Y_test == 1) & (Z_test == 0))));
    
    % EOD
    EOD = max(EOP, ...
        abs((1/size(X1_, 1))*sum(label(Z_test == 1)) -...
        (1/size(X2_, 1))*sum(label(Z_test == 0))));
end

function d = mmd(X1, X2, sigma)
    m = size(X1, 1);
    n = size(X2, 1);
    
    d = (1/m^2) * ones(1,m) * rbf(X1, X1, sigma) * ones(m,1);
    d = d + (1/n^2) * ones(1,n) * rbf(X2, X2, sigma) * ones(n,1);
    d = d - (2/(m*n)) * ones(1,m) * rbf(X1, X2, sigma) * ones(n,1);
end

function K = rbf(X1, X2, sigma)
    K = pdist2(X1, X2, 'squaredeuclidean');
    K = exp(-(1/(2*sigma^2)) * K);
end