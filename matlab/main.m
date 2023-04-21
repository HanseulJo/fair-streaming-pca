clear all;

% rng('default')
%% Create synthetic datasets
% dimensions
d = 40; k = 10; r = 10;
% dim(range(Sigma_gap)) = r
% dim(null(Sigma_gap)) = d - r
p = 0.5;

% conditional means
mu_scale=1;

mu_tmp = randn(d, 1);
mu_tmp = mu_tmp*mu_scale/norm(mu_tmp);
mu0 = mu_tmp * p / (p - 1); mu1  = mu_tmp;
mu = zeros(d, 1);
mu_gap = mu1 - mu0;

% % condition covariance (my approach)
% U = orth(randn(d, d-r));
% J0 = orth(randn(d, r/2));  J1 = orth(randn(d, r));
% Sigma0 = U*U';
% Sigma1 = U*U' + J1*J1';
% Sigma = (1 - p)*Sigma0 + p*Sigma1 + p*(1-p)*(mu_gap*mu_gap');
% Sigma_gap = Sigma1 - Sigma0;
% Sigma_gap = (Sigma_gap + Sigma_gap')/2;

% Hanseul's approach
% conditional covariances
eps = 0.1; max_cov_eig0 = 2; max_cov_eig1 = 2;

% eigenvectors
A = randn(d, r);
W = null([zeros(d,d-r) A]);
W0 = [W orth(A)]; W1 = [W orth(A)];
% eigenvalues
D = eps + (max(max_cov_eig0, max_cov_eig1) - eps)*rand(1,d-r);
D0 = [D eps + (max_cov_eig0 - eps)*rand(1,r)];
D1 = [D 3*eps + (max_cov_eig1 - eps)*rand(1,r)];
% actual generation
Sigma0 = W0 * diag(D0) * W0'; Sigma1 = W1 * diag(D1) * W1';
Sigma = (1 - p)*Sigma0 + p*Sigma1 + p*(1-p)*(mu_gap*mu_gap');
Sigma_gap = Sigma1 - Sigma0;
Sigma_gap = (Sigma_gap + Sigma_gap')/2;
% disp(eig(Sigma_gap))

% figure(100)
% plot(sort(eig(Sigma)))

%% nonfair vanilla PCA
[coeff,~,~,~,~,~] = pca(Sigma);
V_nonfair = coeff(:,1:k);


%% Ground truth fair PCA (ours)
[R, D] = eig(Sigma_gap);
[~,ind] = sort(diag(abs(D)));
R = R(:,ind);
R = R(:,end-r+1:end);
[C, ~] = qr([R mu_gap], "econ");
N = eye(d) - C*C';

% "original" approach (Kleindessner et al., AISTATS 2023)
[coeff,~,~,~,explained,~] = pca(N' * Sigma * N);
V_true = coeff(:,1:k);
% 
% % new approach
% [V_, D] = eig(Sigma);
% [D_, ind] = sort(diag(D));
% V_ = V_(:,ind);
% V_ = V_(:,end-(r+k):end);
% D_ = diag(D_(end-(r+k):end));
% Sigma_ = V_ * D_ * V_';
% [coeff,~,~,~,explained,~] = pca(N' * Sigma_ * N);
% V_true_ = coeff(:,1:k);
% 
% disp(trace(V_true'*Sigma*V_true))
% disp(norm(V_true'*mu_gap))
% disp(norm(V_true'*Sigma_gap*V_true))
% 
% disp(trace(V_true_'*Sigma*V_true_))
% disp(norm(V_true_'*mu_gap))
% disp(norm(V_true_'*Sigma_gap*V_true_))
% 
% disp(norm(V_true - V_true_))
% return

%% run and plot
% (A)NPM: (Accelerated) Noisy Power Method
% FD: with Frequent Directions (Yun, IEEE BigData 2018)
mode = 'NPM'; % NPM, NPMFD, ANPM, ANPMFD

% total iterations
T_R = 10^2;
T = 10^2;

% batch sizes
b = 5*10^3;
B = b;
B0 = b; B1 = b;

trues = {V_true, R, mu_gap, Sigma_gap};
params = {d, k, r, B, B0, B1, mode};
settings = {mu0, mu1, Sigma0, Sigma1, p, Sigma, mu, V_nonfair};

M = {zeros(d,1), zeros(d,1), zeros(d,1)};
n_vec = {0, 0, 0};

% run algorithm
% [R, logs_R] = offline_train_R(params, T_R, trues);
[R, logs_R, M, n_vec] = train_R(M, n_vec, params, T_R, trues, settings);

% [V, logs_V] = offline_train_V(R, params, T, trues, settings);
[V, logs_V, M, n_vec] = train_V(R, M, n_vec, params, T, trues, settings);

% [W, logs_W] = offline_train_W(params, T, trues, settings);
[W, logs_W] = train_W(params, T, trues, settings);


%% plot R
figure(1)
plot(logs_R(1,:))
title("|| R_{true}*R_{true}' - R*R' ||_2")

figure(2)
plot(logs_R(2,:))
title("||(Q-hat(Q))*R||_2")


%% plot V
figure(3)
plot(logs_V(1,:))
hold on
true_var = trace(V_true'*Sigma*V_true)/trace(V_nonfair'*Sigma*V_nonfair);
plot(true_var*ones(1,T))
plot(logs_W(1,:))
plot(ones(1,T))
hold off
title("ratio of explained variance to nonfair PCA")
legend("training V", "V_true", "training W", "W_true")


figure(4)
plot(logs_V(2,:))
hold on
plot(norm(V_true'*mu_gap)*ones(1,T))
plot(logs_W(2,:))
plot(norm(V_nonfair'*mu_gap)*ones(1,T))
hold off
title("projected mean diff")
legend("training V", "V_true", "training W", "W_true")

figure(5)
plot(logs_V(3,:))
hold on
plot(norm(V_true'*Sigma_gap*V_true)*ones(1,T))
plot(logs_W(3,:))
plot(norm(V_nonfair'*Sigma_gap*V_nonfair)*ones(1,T))
hold off
title("projected cov diff")
legend("training V", "V_true", "training W", "W_true")
% 
% figure(6)
% plot(logs(6,:))
% title("||f - f_true||")
