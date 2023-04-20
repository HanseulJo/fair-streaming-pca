import numpy as np
from tqdm import tqdm


class Problem:
    def __init__(self, d, k, p, Sigma0, Sigma1):
        self.d, self.k, self.p = d, k, p
        self.Sigma0, self.Sigma1 = Sigma0, Sigma1

        # covariance difference
        self.Q = self.Sigma0 - self.Sigma1
        # total covariance
        self.Sigma = self.Sigma0 + self.Sigma1

        # Offline Optimal without fairness
        eigenvalues, eigenvectors = np.linalg.eigh(self.Sigma)  # eigenvalues in ascending order
        self.total_var, self.V_star = sum(eigenvalues[-k:]), eigenvectors[:, -k:]

    def sample(self):
        s = np.random.binomial(1, self.p)  # biased choice of binary attribute
        _Sigma = [self.Sigma0, self.Sigma1][s]
        x = np.random.multivariate_normal(np.zeros(self.d), _Sigma, 1).T
        return s, x

    def solve_offline(self, V0, rho, N, alpha0=1e-3):
        if V0 is None:
            V = np.random.rand(self.d, self.k)
            V, _ = np.linalg.qr(V)
        else:
            V = V0.copy()

        alpha = alpha0
        history = [V]
        for n in tqdm(range(1, N + 1)):
            #if n % (N//10) == 0:
            #    alpha *= 0.5
            G = self.Sigma @ V
            tilde_G = self.Q @ V
            s = np.sign(np.trace(self.Q) - np.trace(V.T @ tilde_G))
            V, _ = np.linalg.qr(V + alpha * (G + s * rho * tilde_G))
            # V, _ = np.linalg.qr(V + alpha * s * tilde_G)
            history.append(V)
        return history

    def solve_offline_Q(self, alpha0=1e-3):
        N = int(1e5)

        # V = self.V_star
        V = np.random.rand(self.d, self.k)
        V, _ = np.linalg.qr(V, mode='reduced')
        alpha = alpha0
        history = [V]
        for n in tqdm(range(1, N + 1)):
            tilde_G = self.Q @ V
            s = np.sign(np.trace(self.Q) - np.trace(V.T @ tilde_G))
            V, _ = np.linalg.qr(V + alpha * s * tilde_G, mode='reduced')
            history.append(V)
        return history