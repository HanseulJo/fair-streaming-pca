function [V, logs_V] = offline_train_V(R, params, T, trues, settings)
    % unpack parameters
    [d, k, ~, ~, ~, ~, mode] = params{:};
    [V_true, ~, mu_gap, Sigma_gap] = trues{:};
    [~, ~, ~, ~, ~, Sigma, ~, V_nonfair] = settings{:};
    
    [C, ~] = qr([R mu_gap], "econ");
    N = eye(d) - C*C';
    Sigma_R = N' * Sigma * N;

    logs_V = zeros(4, T);

    [V, L] = qr(randn(d, k), "econ");
    V_prev = zeros(d, k);
    beta = 0;
    % acceleration
    if contains(mode, "ANPM")
        beta = 0.01;
    end

    for t = 1:T
        [V, L] = qr(Sigma_R*V - beta*V_prev*inv(L), "econ");
        % logging
        logs_V(1,t) = trace(V'*Sigma*V) / trace(V_nonfair'*Sigma*V_nonfair);
        logs_V(2,t) = norm(V'*mu_gap, 'fro');
        logs_V(3,t) = norm(V'*Sigma_gap*V, 'fro');
        logs_V(4,t) = norm(V_true - V, 'fro');
    end
end