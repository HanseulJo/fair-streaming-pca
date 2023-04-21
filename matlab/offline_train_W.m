function [W, logs_W] = offline_train_W(params, T, trues, settings)
    % unpack parameters
    [d, k, ~, ~, ~, ~, mode] = params{:};
    [~, ~, mu_gap, Sigma_gap] = trues{:};
    [~, ~, ~, ~, ~, Sigma, ~, V_nonfair] = settings{:};
    
    logs_W = zeros(4, T);

    [W, L] = qr(randn(d, k), "econ");
    W_prev = zeros(d, k);
    beta = 0;
    % acceleration
    if contains(mode, "ANPM")
        beta = 0.01;
    end

    for t = 1:T
        [W, L] = qr(Sigma*W - beta*W_prev*inv(L), "econ");
        % logging
        logs_W(1,t) = trace(W'*Sigma*W) / trace(V_nonfair'*Sigma*V_nonfair);
        logs_W(2,t) = norm(W'*mu_gap, 'fro');
        logs_W(3,t) = norm(W'*Sigma_gap*W, 'fro');
        logs_W(4,t) = norm(V_nonfair - W, 'fro');
    end
end