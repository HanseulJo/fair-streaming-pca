function [R, logs_R, M, n_vec] = train_R(M, n_vec, params, T_R, trues, settings)
    % unpack parameters
    [~, R_true, ~, Sigma_gap] = trues{:};
    [d, ~, r, ~, B0, B1, mode] = params{:};

    [mu, mu0, mu1] = M{:}; [n, n0, n1] = n_vec{:};

    logs_R = zeros(2, T_R);

    [R, L] = qr(randn(d, r), "econ");
    R_prev = zeros(d, r);
    beta = 0;

    % acceleration
    if contains(mode, "ANPM")
        beta = 0.01;
    end

    for t = 1:T_R
        m0 = 0; m1 = 0;
        G0 = zeros(d, r); G1 = zeros(d, r);
        while m0 < B0 || m1 < B1
            [s, x] = environment_sample(settings);
            % update mu's & compute \hat{Q}R
            mu = (n / (n+1)) * mu + (1 / (n+1)) * x;
            n = n + 1;
            if s == 0
                mu0 = (n0 / (n0+1)) * mu0 + (1 / (n0+1)) * x;
                n0 = n0 + 1;

                G0 = G0 + (x - mu0)*(x - mu0)'*R;
                m0 = m0 + 1;
            else
                mu1 = (n1 / (n1+1)) * mu1 + (1 / (n1+1)) * x;
                n1 = n1 + 1;

                G1 = G1 + (x - mu1)*(x - mu1)'*R;
                m1 = m1 + 1;
            end
        end
        G = (1/m1)*G1 - (1/m0)*G0;
        
        [R, L] = qr(G - beta*R_prev*inv(L), "econ");
        % logging
        logs_R(1,t) = norm(R_true*R_true' - R*R');
        logs_R(2,t) = norm(Sigma_gap*R-G);
    end

    M = {mu, mu0, mu1}; n_vec = {n, n0, n1};
end