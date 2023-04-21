function [s, x] = environment_sample(settings)
    [mu0, mu1, Sigma0, Sigma1, p, ~, ~, ~] = settings{:};
    [d, ~] = size(mu0);
    s = binornd(1, p);
    if s == 0
        x = mvnrnd(mu0, Sigma0 + 0.01*eye(d));
    else
        x = mvnrnd(mu1, Sigma1 + 0.01*eye(d));
    end
    x = x';
end