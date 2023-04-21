function [R, logs_R] = offline_train_R(params, T_R, trues)
    % unpack parameters
    [d, ~, r, ~, ~, ~, mode] = params{:};
    [~, R_true, ~, Sigma_gap] = trues{:};

    logs_R = zeros(2, T_R);

    [R, L] = qr(randn(d, r), "econ");
    R_prev = zeros(d, r);
    beta = 0;

    % acceleration
    if contains(mode, "ANPM")
        beta = 0.01;
    end

    for t = 1:T_R
        [R, L] = qr(Sigma_gap*R - beta*R_prev*inv(L), "econ");
        % logging
        logs_R(1,t) = norm(R_true*R_true' - R*R', 'fro');
    end
end