function [V, logs_V, M, n_vec] = train_V(R, M, n_vec, params, T, trues, settings)
    % unpack parameters
    [d, k, ~, B, ~, ~, mode] = params{:};
    [V_true, ~, mu_gap, Sigma_gap] = trues{:};
    [~, ~, ~, ~, ~, Sigma, ~, V_nonfair] = settings{:};

    [mu, mu0, mu1] = M{:}; [n, n0, n1] = n_vec{:};

    logs_V = zeros(4, T);

    [V, L] = qr(randn(d, k), "econ");
    V_prev = zeros(d, k);
    beta = 0;
    % acceleration
    if contains(mode, "ANPM")
        beta = 0.01;
    end

    for t = 1:T
        f = mu1 - mu0;
        [C, ~] = qr([R f], "econ");
        N = eye(d) - C*C';
        
        m0 = 0; m1 = 0;
        G = zeros(d, k);
        while m0 + m1 < B
            [s, x] = environment_sample(settings);
            % update mu's & compute \hat{Q}R
            mu = (n / (n+1)) * mu + (1 / (n+1)) * x;
            n = n + 1;
            if s == 0
                mu0 = (n0 / (n0+1)) * mu0 + (1 / (n0+1)) * x;
                n0 = n0 + 1;
                m0 = m0 + 1;
            else
                mu1 = (n1 / (n1+1)) * mu1 + (1 / (n1+1)) * x;
                n1 = n1 + 1;
                m1 = m1 + 1;
            end
            y = N*x;
            G = G + y*y'*V;
        end
        
        [V, L] = qr(G - beta*V_prev*inv(L), "econ");
        % logging
        logs_V(1,t) = trace(V'*Sigma*V) / trace(V_nonfair'*Sigma*V_nonfair);
        logs_V(2,t) = norm(V'*mu_gap, 'fro');
        logs_V(3,t) = norm(V'*Sigma_gap*V, 'fro');
        logs_V(4,t) = norm(V_true - V, 'fro');
    end
    
    M = {mu, mu0, mu1}; n_vec = {n, n0, n1};
end