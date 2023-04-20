from argparse import ArgumentParser
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as np
from scipy.linalg import null_space
from typing import Iterable, Tuple
from tqdm.auto import trange
# from sklearn.covariance import shrunk_covariance, ledoit_wolf, oas


def cos(A:np.ndarray, B:np.ndarray):
    assert A.shape == B.shape
    return np.sum(A*B) / np.sqrt(np.square(A).sum() * np.square(B).sum())


def grassmanian_distance(A:np.ndarray, B:np.ndarray):
    """
    distance between matrix A and B with same size,
    invariant to column ordering
    """
    assert A.shape == B.shape
    assert A.ndim <= 2
    if A.ndim == 1:
        return np.linalg.norm(np.outer(A,A)-np.outer(B,B))
    assert A.shape[0] >= A.shape[1]
    return np.linalg.norm(A @ A.T - B @ B.T)


def truncate_rows(A:np.ndarray, leave=None):
    assert A.ndim == 2, A.ndim
    d, k = A.shape
    if leave is None: leave = k
    if d > k:
        norms = np.linalg.norm(A, ord=2, axis=0)
        # A[np.argsort(norms)[:d-leave]] = 0
        A = A.at[np.argsort(norms)[:d-leave]].set(0)
    return A

    
class Buffer:
    """
    Arbitrary Memory buffer to record performance of PCA algorithm.
    """
    def __init__(self):
        self.pca = []                        # V matrices
        self.explained_variance_ratio = []   # objective function, [0, 1]
        self.distance_from_gt = []           # Grassmanian distance of V and V_ground_truth
        self.objective_unfairness = []       # difference btw group-conditional exp_var_ratio
        self.R_dist = []                     # Grassmanian distance of R and R_opt
        self.projected_R_norm = []          # trace(R' * Sigma_gap * R)
        self.projected_mean_diff = []        # intensity of mean equalization
        self.projected_cov_diff = []        # intensity of mean equalization

class StreamingFairBlockPCA:
    """
    Streaminig Fair Block PCA on a synthetic multivariate-normal distributions.
    """
    
    def __init__(self,
            data_dim,
            num_attributes=1,
            num_groups=2,
            probability=0.5,
            seed=None,
            nullity=None,
            mu_scale = 1.,
            eps = 0.1,
            max_cov_eig0 = 2.,
            max_cov_eig1 = 1.,
        ) -> None:
        """
        Initialization of synthetic distrubutions.

        :: Input
            - data_dim (int)
            - num_attribute (int)
            - num_groups (int)
            - probability (float or Iterable)
            - seed (int, optional)
            - rank (int, optional)
            - mu_scale (float, optional) : default = 1.

        :: Notes
            - Not implemented for `num_attribute` > 1 yet
            - Not implemented for `num_groups` > 2 yet
            - Not implemented for `Iterable` type `probability` yet
        """
        ## Check arguments
        assert isinstance(data_dim, int) and data_dim >= 2, f"Invalid data_dim: {data_dim}"
        assert isinstance(num_attributes, int) and num_attributes >= 1, f"Invalid num_attributes: {num_attributes}"
        assert isinstance(num_groups, int) and num_groups >= 2, f"Invalid num_groups: {num_groups}"
        assert isinstance(probability, (float)), f"Invalid probability: {probability}"
        if isinstance(probability, float):
            assert 0 < probability < 1, f"Invalid probability: {probability}"
            assert num_attributes == 1 and num_groups == 2, f"Requires binary-group single-attribute fairness if `probability` is a float"
        else: #  isinstance(probability, Iterable):
            raise NotImplementedError
        if nullity is not None:
            assert isinstance(nullity, int) and 0 <= nullity <= data_dim, f"Invalid rank: {nullity}"

        self.d = data_dim
        self.num_attr = num_attributes
        self.num_groups = num_groups
        self.p = probability
        self._r = nullity if nullity is not None else (data_dim // 2)

        ## Random generator
        # rng = np.random.default_rng(seed) 
        if seed is None:
            import numpy as _np
            from sys import maxsize
            seed = _np.random.randint(0,maxsize)
        key = random.PRNGKey(seed)

        ## Conditional mean vectors:
        ## (1-p) * mu0 + p * mu1 = 0  (Assuming true mean == zero vector)
        # mu_temp = rng.standard_normal(self.d)
        key, rng = random.split(key)
        mu_temp = random.normal(rng, (self.d,))
        mu_temp *= mu_scale / np.linalg.norm(mu_temp) 
        self.mu0 = mu_temp * self.p / (self.p - 1.)
        self.mu1 = mu_temp
        self.mu = (1-self.p) * self.mu0 + self.p * self.mu1
        self.mu_gap = self.mu1 - self.mu0   # f
        assert np.isclose(np.abs(self.mu).max(), 0.), f"mu is not close to 0."
        
        ## Conditional covariances:
        ### 1. Orthogonal matrices W0 & W1 (sharing r same columns)
        dim_gap = self.d-self._r
        # A_ = rng.standard_normal((self.d, self._r))
        # A0 = np.concatenate([A_, rng.standard_normal((self.d, dim_gap))], axis=1)
        # A1 = np.concatenate([A_, rng.standard_normal((self.d, dim_gap))], axis=1)
        key, rng = random.split(key)
        A_ = random.normal(rng, (self.d, self._r))
        key, rng = random.split(key)
        A0 = np.concatenate([A_, random.normal(rng, (self.d, dim_gap))], axis=1)
        key, rng = random.split(key)
        A1 = np.concatenate([A_, random.normal(rng, (self.d, dim_gap))], axis=1)
        W0, _ = np.linalg.qr(A0)   # orthogonal
        W1, _ = np.linalg.qr(A1)   # orthogonal
        if self._r > 0:
            assert np.isclose(np.max(W0[:,:self._r] - W1[:,:self._r]), 0.), f"Invalid orthogonal matrices: W0[:,:r] != W1[:,:r]"  # This command also checks rank validity
        
        ### 2. Eigenvalues D0 & D1 (sharing r same eigenvalues)
        # D_ = rng.normal(0, max(max_cov_eig0, max_cov_eig1)*3, self._r)
        # D0 = np.concatenate([D_, rng.normal(0, max_cov_eig0, dim_gap)])
        # D1 = np.concatenate([D_, rng.normal(0, max_cov_eig1, dim_gap)])
        key, rng = random.split(key)
        D_ = random.normal(rng, (self._r,)) * (max(max_cov_eig0, max_cov_eig1)*3)
        key, rng = random.split(key)
        D0 = np.concatenate([D_, random.normal(rng, (dim_gap,)) * max_cov_eig0])
        key, rng = random.split(key)
        D1 = np.concatenate([D_, random.normal(rng, (dim_gap,)) * max_cov_eig1])
        # D0[:self._r] += rng.uniform(-eps, eps, self._r)
        D0, D1 = np.abs(D0) ** 1, np.abs(D1) ** 1

        ### 3. Eigen-Composition to make Sigma0 & Sigma1;
        ###     rank(Sigma1 - Sigma0) <= d - r.
        self.Sigma0 = W0 @ np.diag(D0) @ W0.T
        self.Sigma1 = W1 @ np.diag(D1) @ W1.T
        self.Sigma = (1-self.p) * self.Sigma0 + self.p * self.Sigma1 + self.p*(1-self.p) * np.outer(self.mu_gap, self.mu_gap)
        assert np.min(np.linalg.eigh(self.Sigma)[0]) >= 0, f"Invalid Sigma: non-PSD, {np.linalg.eigh(self.Sigma)[0]}"
        self.Sigma_gap = self.Sigma1 - self.Sigma0
        
        ## Info about gaps
        # rank_Sigma_gap = np.linalg.matrix_rank(self.Sigma_gap, hermitian=True)
        self.trace_Sigma_gap = np.trace(self.Sigma_gap)
        self.Sigma_gap_sq = self.Sigma_gap @ self.Sigma_gap
        self.eigval_Sigma_gap_sq, self.eigvec_Sigma_gap_sq = np.linalg.eigh(self.Sigma_gap_sq)  # eigenvalue of Q^2 increasing order
        # assert rank_Sigma_gap == dim_gap, f"Invalid rank(Sigma_gap)={rank_Sigma_gap} (> {dim_gap}=d-r)"  # This command checks rank validity 
        
        ## Optima / group-conditional optima
        self.eigval_Sigma, self.eigvec_Sigma = np.linalg.eigh(self.Sigma)  # eigenvalues in ascending order
        # self.total_var, self.V_star = self.eigval_Sigma.sum(), self.eigvec_Sigma
        self.group_vars, self.group_V_stars = [0]*num_groups, [0]*num_groups
        for s in range(self.num_groups):
            eigenvalues, eigenvectors = np.linalg.eigh(eval(f"self.Sigma{s}"))
            self.group_vars[s] = np.sum(eigenvalues)
            self.group_V_stars[s] = eigenvectors
    

    def get_ground_truth(self,
            target_dim,
            rank=None,
            mode='all'  # ['mean', 'covariance', 'all']
            ):
        """
        Given k (`target_dim`) and r (`rank`), 
        get groud-truth fair PC matrix.

        :: Input
            - target_dim (int)
            - rank (int, optional) : if none, only mean matching will occur
            - mode (str) : ['mean', 'covariance', 'all']
        :: Return
            - explained_variance (float)
            - groud_truth (np.ndarray) : size=(data_dim, target_dim)
        """
        assert isinstance(target_dim, int) and 0 < target_dim < self.d, f"Invalid target_dim: {target_dim}"
        g = self.mu_gap / np.linalg.norm(self.mu_gap)


        if mode == 'mean':
            R = null_space(g.reshape(1,-1))
        elif mode == 'covariance':
            R = self.eigvec_Sigma_gap_sq[:,:self.d-rank]
        elif mode == 'all': 
            N = self.eigvec_Sigma_gap_sq[:,-rank:]
            N = np.concatenate([g.reshape(-1,1), N], 1)
            N, _ = np.linalg.qr(N)
            RR = np.eye(self.d) - N @ N.T
            D, Q = np.linalg.eigh(RR)
            R = Q @ np.diag(np.sqrt(np.abs(D)))
        else:
            return self.eigval_Sigma[-target_dim:].sum(), self.eigvec_Sigma[:,-target_dim:]
        M = R.T @ self.Sigma @ R
        eigval, eigvec = np.linalg.eigh(M)
        explained_variance = np.sum(eigval[-target_dim:])
        ground_truth = R @ eigvec[:,-target_dim:]
        # if mode in ['mean', 'all']:
        #     assert np.max(np.abs(ground_truth.T @ self.mu_gap)) < 1e-6, \
        #         f"Failed to equalizing means, {np.max(np.abs(ground_truth.T @ self.mu_gap))}"
        return explained_variance, ground_truth
    
    
    def get_explained_variance_ratio(self, V:np.ndarray=None, VVT:np.ndarray=None):
        """
        Given V, get explained_variance_ratio (== total objective function)

        :: Input
            - V (np.ndarray) : size=(data_dim, target_dim)

        :: Return
            - exp_var (float)
        """
        assert hasattr(self, 'total_var')
        if VVT is not None:
            return np.trace(self.Sigma @ VVT) / self.total_var
        else:
            return np.trace(self.Sigma @ V @ V.T) / self.total_var


    def get_group_explained_variance_ratio(self, s, V:np.ndarray=None, VVT:np.ndarray=None):
        """
        Given V, get group-conditional explained_variance

        :: Input
            - V (np.ndarray) : size=(data_dim, target_dim)

        :: Return
            - exp_var (float)
        """
        assert isinstance(s, int) and 0 <= s < self.num_groups
        assert hasattr(self, 'group_vars')
        if VVT is not None:
            return np.trace(eval(f"self.Sigma{s}") @ VVT) / self.group_vars[s]
        else:
            return np.trace(eval(f"self.Sigma{s}") @ V @ V.T) / self.group_vars[s]
    

    def get_objective_unfairness(self, V:np.ndarray=None, VVT:np.ndarray=None):
        """
        Given V, get objective unfairness (difference in conditional objective function)

        :: Input
            - V (np.ndarray) : size=(data_dim, target_dim)

        :: Return
            - exp_var (float)
        """
        exp_var_ratio = self.get_explained_variance_ratio(V=V, VVT=VVT)
        exp_var_group_ratios = [self.get_group_explained_variance_ratio(s, V=V, VVT=VVT) for s in range(self.num_groups)]
        return max(abs(exp_var_ratio - exp_var_group_ratios[s]) for s in range(self.num_groups))


    def evaluate(self, R:np.ndarray=None, V:np.ndarray=None, rank=None, mode='vanilla'):
        if R is not None:
            r = R.shape[1]
            R_true = self.eigvec_Sigma_gap_sq[:,-r:]
            self.buffer.R_dist.append(grassmanian_distance(R, R_true))
            # self.buffer.projected_R_norm.append(np.linalg.norm(R.T @ self.Sigma_gap @ R))
        if V is not None:
            VVT = V @ V.T
            k = V.shape[1]
            if not hasattr(self, 'exp_var_gt'):
                if rank is None and hasattr(self, 'r'): rank = self.r
                self.exp_var_gt, self.V_ground_truth = self.get_ground_truth(k, rank, mode=mode)
            self.buffer.explained_variance_ratio.append(self.get_explained_variance_ratio(VVT=VVT))
            self.buffer.objective_unfairness.append(self.get_objective_unfairness(VVT=VVT))
            if self.V_ground_truth is not None:
                self.buffer.distance_from_gt.append(np.linalg.norm(self.V_ground_truth @ self.V_ground_truth.T - VVT))
            self.buffer.projected_mean_diff.append(np.linalg.norm(self.mu_gap.T @ V))
            self.buffer.projected_cov_diff.append(np.linalg.norm(V.T @ self.Sigma_gap @ V))


    def offline_train(self,
            target_dim,
            rank,
            n_iter,
            lr,
            seed=None,
            tol=None,
            mode='oja',         # ['oja', 'pm']
            constraint='all',   # ['vanilla', 'mean', 'covariance', 'all']
            lr_scheduler=None,
            landing_lambda = .1,
        ):
        """
        train V in offline manner with fairness constraint
        """
        ## Check arguments
        assert isinstance(target_dim, int) and target_dim < self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert isinstance(lr, (int, float)) and lr > 0
        assert mode in ['oja', 'pm', 'riemannian', 'landing']
        assert constraint in ['vanilla', 'mean', 'covariance', 'all']

        # rng = np.random.default_rng(seed)
        if seed is None:
            import numpy as _np
            from sys import maxsize
            seed = _np.random.randint(0,maxsize)
        key = random.PRNGKey(seed)
        self.n_iter = n_iter
        self.k = target_dim
        if constraint in ['covariance', 'all']:
            self.r = rank
        
        ## Buffers
        self.buffer = Buffer()
        self.total_var, self.V_star = self.get_ground_truth(self.k, rank, mode=None)
        self.exp_var_gt, self.V_ground_truth = self.get_ground_truth(self.k, rank, mode=constraint)

        if constraint in ['mean', 'all']:
            ## Mean gap
            f = self.mu_gap / np.linalg.norm(self.mu_gap)
            N = f.reshape(-1, 1)   # this line is used only when `constrain == 'mean'`
        if constraint in ['covariance','all']:
            ## Covariance gap
            R_ = self.eigvec_Sigma_gap_sq[:,-rank:]
            N = R_   # this line is used only when `constrain == 'covariance'`
        if constraint == 'all':
            ## Normal subspace; for both mean & covariance gap
            N = np.concatenate([f.reshape(-1,1), R_], 1)
            N, _ = np.linalg.qr(N)
        # V, _ = np.linalg.qr(rng.standard_normal((self.d, self.k)))
        key, rng = random.split(key)
        V, _ = np.linalg.qr(random.normal(rng, (self.d, self.k)))
        
        self.evaluate(V=V)
        
        lr0 = lr
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _V = V.copy()  # store previous V
            if lr_scheduler is not None:
                lr = lr0 * lr_scheduler(t)
            ## Gradient
            if constraint == 'vanilla': G = self.Sigma @ V
            else: 
                G = self.Sigma @ (V - N @ N.T @ V)
                G -= N @ N.T @ G
            ## Update
            if mode == 'oja':   V += lr * G
            elif mode == 'pm':  V = G
            elif mode == 'riemannian': 
                riemannian_grad = .5 * (G - V @ (G.T @ V))
                V += lr * riemannian_grad
            elif mode == 'landing':
                riemannian_grad = .5 * (G - V @ (G.T @ V))
                field = V @ (V.T @ V) - V
                V += lr * (riemannian_grad + landing_lambda * field)
            if mode != 'landing':
                V, _ = np.linalg.qr(V)
            
            self.evaluate(V=V)
            if tol is not None:
                _gr = grassmanian_distance(_V, V)  # compare Lam
                pbar.set_description(f"gap={_gr:.6f}")
                if _gr < tol:
                    print(f"OUTER: Broke in {t} / {n_iter}"); break
        return V
    

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Receive a pair of data `x` and corresponding sensitive attribute `s`

        :: Input
            - key (jax.numpy.PRNGKeyArray, optional)

        :: Return
            - s (int) : {0, 1}
            - x (np.ndarray) : size=(data_dim, 1)
        """
        if hasattr(self, 'key'):
            # s = rng.binomial(1, self.p)
            # x = rng.multivariate_normal(eval(f"self.mu{s}"), eval(f"self.Sigma{s}"))
            self.key, rng = random.split(self.key)
            s = random.bernoulli(rng, self.p).astype(int)
            self.key, rng = random.split(self.key)
            x = random.multivariate_normal(rng, eval(f"self.mu{s}"), eval(f"self.Sigma{s}"))
        else:
            import numpy as _np
            s = _np.random.binomial(1, self.p)
            x = _np.random.multivariate_normal(eval(f"self.mu{s}"), eval(f"self.Sigma{s}"))
        return s, x


    def train(self,
            target_dim,
            rank,
            n_iter,
            batch_size,
            constraint='all',             # ['vanilla', 'mean', 'covariance', 'all']
            mean_estimation='local',      # ['local', 'global']
            subspace_optimization='oja',  # ['oja', 'npm', 'riemannian', 'landing', 'history']
            pca_optimization='npm',       # ['oja', 'npm', 'riemannian', 'landing', 'history']
            lr_subspace=None,             # must be float when `subspace_optimization != 'npm'`
            lr_pca=None,                  # must be float when `pca_optimization != 'npm'`
            n_iter_history=None,            # must be int when `subspace_optimization in ['history'] or pca_optimization in ['history']`
            landing_lambda=None,          # must be float when `subspace_optimization == 'landing'`
            seed=None,
            tol=None,
            lr_scheduler=None,
        ):
        """
        Streaminig Fair Block PCA Algorithm

        """
        ## Check arguments
        assert isinstance(target_dim, int) and 0 < target_dim < self.d
        assert isinstance(rank, int) and 0 < rank <= self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert isinstance(batch_size, int) and batch_size > 0
        assert constraint in ['vanilla', 'mean', 'covariance', 'all'], constraint
        assert mean_estimation in ['local', 'global'], mean_estimation
        if constraint != 'vanilla':
            assert subspace_optimization in ['oja', 'npm', 'npmfd', 'riemannian', 'landing', 'history'], subspace_optimization
        assert pca_optimization in ['oja', 'npm', 'npmfd', 'riemannian', 'landing', 'history'], pca_optimization
        if subspace_optimization not in ['npm', 'npmfd', 'history'] and constraint != 'vanilla':
            assert isinstance(lr_subspace, (int, float)) and lr_subspace > 0
        if pca_optimization not in ['npm', 'npmfd', 'history'] :
            assert isinstance(lr_pca, (int, float)) and lr_pca > 0
        if subspace_optimization in ['history'] or pca_optimization in ['history']:
            assert isinstance(n_iter_history, int) and n_iter_history > 0
        if subspace_optimization in ['landing']:
            assert isinstance(landing_lambda, float) and landing_lambda > 0

        if seed is None:
            import numpy as _np
            from sys import maxsize
            seed = _np.random.randint(0,maxsize)
        key = random.PRNGKey(seed)
        self.n_iter = n_iter
        self.k = target_dim
        if constraint in ['covariance', 'all']:
            self.r = rank

        ## Buffers
        self.buffer = Buffer()
        self.exp_var_gt, self.V_ground_truth = self.get_ground_truth(self.k, rank, mode=constraint)
        
        R = None
        if constraint in ['covariance', 'all']:
            key, rng = random.split(key)
            R, _ = np.linalg.qr(random.normal(rng, (self.d, rank)))
        key, rng = random.split(key)
        V, _ = np.linalg.qr(random.normal(rng, (self.d, self.k)))
        self.evaluate(R=R,V=V)

        n_global = [0 for _ in range(self.num_groups)]  # n per group
        group_mean_global = [np.zeros(self.d) for _ in range(self.num_groups)]
        if subspace_optimization == 'npmfd': B_R = np.zeros((self.d, 2*rank))
        if pca_optimization == 'npmfd': B_V = np.zeros((self.d, 2*self.k))
        if subspace_optimization == 'history': D_R = np.zeros((rank, rank))
        if pca_optimization == 'history': D_V = np.zeros((self.k, self.k))
        lr_pca0, lr_subspace0 = lr_pca, lr_subspace

        pbar = trange(1, n_iter+1)
        for t in pbar:
            _V = V.copy()       # store previous V
            if R is not None: _R = R.copy()       # store previous R
            if lr_scheduler is not None and lr_subspace is not None: lr_subspace = lr_subspace0 * lr_scheduler(t)
            if lr_scheduler is not None and lr_pca is not None: lr_pca = lr_pca0 * lr_scheduler(t)
            
            ## Sampling
            n_local1 = [0 for _ in range(self.num_groups)]  # n per group, local
            A1, X1 = [], []
            # while min(n_local1)<batch_size and not (sum(n_local1)>=2*batch_size and min(n_local1)>0):
            while min(n_local1)<batch_size:
                s, x = self.sample()
                A1.append(s)
                X1.append(x)
                n_local1[s] += 1
            for s in range(self.num_groups):
                n_global[s] += n_local1[s]
            A1 = np.array(A1)
            X1 = np.stack(X1, 0)  # (batch_size, dim)
            
            ## Sampling
            n_local2 = [0 for _ in range(self.num_groups)]  # n per group, local
            A2, X2 = [], []
            # while min(n_local2)<batch_size and not (sum(n_local2)>=2*batch_size and min(n_local2)>0):
            while min(n_local2)<batch_size:
                s, x = self.sample()
                A2.append(s)
                X2.append(x)
                n_local2[s] += 1
            for s in range(self.num_groups):
                n_global[s] += n_local2[s]
            A2 = np.array(A2)
            X2 = np.stack(X2, 0)  # (batch_size, dim)

            n_local = [n1+n2 for n1, n2 in zip(n_local1, n_local2)]
            
            group_mean_local = [np.zeros(self.d) for _ in range(self.num_groups)]
            ## group-mean calculation
            for s in range(self.num_groups):
                group_mean_local[s] = np.mean(np.concatenate([X1, X2], 0)[np.concatenate([A1, A2])==s], 0)
                group_mean_global[s] = group_mean_global[s] * (n_global[s] - n_local[s]) / n_global[s] + (n_local[s]/n_global[s]) * group_mean_local[s]
            if constraint == 'vanilla':
                # ## centering
                # if mean_estimation == 'local':
                #     X1 -= np.concatenate([X1, X2], 0).mean(0)
                #     X2 -= np.concatenate([X1, X2], 0).mean(0)
                # if mean_estimation == 'global':
                #     mean_global = sum((n_global[s]/sum(n_global)) * group_mean_local[s] for s in range(self.num_groups))
                #     X1 -= mean_global
                #     X2 -= mean_global
                pass
            else:
                ## group-wise centering
                for s in range(self.num_groups):
                    if mean_estimation == 'local':
                        X1 = X1.at[A1==s].set(X1[A1==s] - group_mean_local[s])
                        X2 = X2.at[A2==s].set(X2[A2==s] - group_mean_local[s])
                    if mean_estimation == 'global':
                        X1 = X1.at[A1==s].set(X1[A1==s] - group_mean_global[s])
                        X2 = X2.at[A2==s].set(X2[A2==s] - group_mean_global[s])
                if constraint in ['mean', 'all']:
                    ## Mean gap
                    f = group_mean_global[1] - group_mean_global[0]
                    f /= np.linalg.norm(f)
                    N = f.reshape(-1, 1)   # this line is used only when `constrain == 'mean'`
                if constraint in ['covariance', 'all']:
                    ## Covariance gap (squared)
                    def Q_hat_squared_R(X1,X2,R):
                        group_covR1 = [np.zeros((self.d, self.r)) for _ in range(self.num_groups)]
                        group_covR2 = [np.zeros((self.d, self.r)) for _ in range(self.num_groups)]
                        group_covcovR1 = [np.zeros((self.d, self.r)) for _ in range(self.num_groups)]
                        group_covcovR2 = [np.zeros((self.d, self.r)) for _ in range(self.num_groups)]
                        for s in range(self.num_groups):
                            group_covR1[s] = X1[A1==s].T @ (X1[A1==s] @ R) / n_local1[s]
                            group_covR2[s] = X2[A2==s].T @ (X2[A2==s] @ R) / n_local2[s]
                        covdiff_R1 = group_covR1[1] - group_covR1[0]
                        covdiff_R2 = group_covR2[1] - group_covR2[0]
                        for s in range(self.num_groups):
                            group_covcovR1[s] = X1[A1==s].T @ (X1[A1==s] @ covdiff_R2) / n_local1[s]
                            group_covcovR2[s] = X2[A2==s].T @ (X2[A2==s] @ covdiff_R1) / n_local2[s] 
                        covdiff_covdiff_R1 = group_covcovR1[1] - group_covcovR1[0]  # Q_hat_1 * Q_hat_2 * R
                        covdiff_covdiff_R2 = group_covcovR2[1] - group_covcovR2[0]  # Q_hat_2 * Q_hat_1 * R
                        covdiff_covdiff_R = (covdiff_covdiff_R1 + covdiff_covdiff_R2) / 2
                        return covdiff_covdiff_R
                    
                    if subspace_optimization == 'oja': 
                        R += lr_subspace * Q_hat_squared_R(X1,X2,R)
                        R, _ = np.linalg.qr(R)
                    elif subspace_optimization == 'npm': 
                        R = Q_hat_squared_R(X1,X2,R)
                        R, _ = np.linalg.qr(R)
                    elif subspace_optimization == 'npmfd': 
                        R = Q_hat_squared_R(X1,X2,R)
                        R, _ = np.linalg.qr(R)
                        B_R = B_R.at[:,-self.r:].set(R)
                        U_R, S_R, _ = np.linalg.svd(B_R, full_matrices=False)  # singular value in decreasing order
                        min_singular = S_R[self.r] ** 2
                        S_R = np.sqrt(np.clip(np.square(S_R)-min_singular, 0, None))
                        B_R = U_R @ np.diag(S_R)
                        R = B_R[:,:self.r]
                        R, _ = np.linalg.qr(R)
                    elif subspace_optimization == 'riemannian':
                        covdiff_covdiff_R = Q_hat_squared_R(X1,X2,R)
                        riemannian_grad = .5 * (covdiff_covdiff_R - R @ (covdiff_covdiff_R.T @ R))
                        R += lr_subspace * riemannian_grad
                        R, _ = np.linalg.qr(R)
                    elif subspace_optimization == 'landing':
                        covdiff_covdiff_R = Q_hat_squared_R(X1,X2,R)
                        riemannian_grad = .5 * (covdiff_covdiff_R - R @ (covdiff_covdiff_R.T @ R))
                        field = R @ (R.T @ R) - R
                        R += lr_subspace * (riemannian_grad + landing_lambda * field)
                    elif subspace_optimization == 'history':
                        for m in range(1,n_iter_history+1): 
                            S = (n_local[s]/n_global[s]) * Q_hat_squared_R(X1,X2,R)  \
                                + ((n_global[s] - n_local[s])/n_global[s]) * _R @ D_R @ _R.T @ R
                            R, _ = np.linalg.qr(S)
                            D_R = np.diag(np.linalg.norm(S, ord=2, axis=0))

                    N = R   # this line is used only when `constrain == 'covariance'`
                if constraint == 'all':
                    ## Normal subspace; for both mean & covariance gap
                    N = np.concatenate([f.reshape(-1,1), R], 1)
                    N, _ = np.linalg.qr(N)
            
            ## Gradient
            def gradient(X1,X2,V):
                if constraint == 'vanilla':  G = (X1.T @ X1 @ V + X2.T @ X2 @ V) / sum(n_local)
                else:
                    G = (X1.T @ X1 @ (V - N @ N.T @ V) + X2.T @ X2 @ (V - N @ N.T @ V)) / sum(n_local)
                    G -= N @ N.T @ G
                return G
            ## Update
            if pca_optimization == 'oja':
                V += lr_pca * gradient(X1,X2,V)
                V, _ = np.linalg.qr(V)
            elif pca_optimization == 'npm':
                V = gradient(X1,X2,V)
                V, _ = np.linalg.qr(V)
            elif pca_optimization == 'npmfd':
                V = gradient(X1,X2,V)
                V, _ = np.linalg.qr(V)
                B_V = B_V.at[:,-self.k:].set(V)
                U_V, S_V, _ = np.linalg.svd(B_V, full_matrices=False)  # singular value in decreasing order
                min_singular = S_V[self.k] ** 2
                S_V = np.sqrt(np.clip(np.square(S_V)-min_singular, 0, None))
                B_V = U_V @ np.diag(S_V)
                V = B_V[:,:self.k]
                V, _ = np.linalg.qr(V)
            elif pca_optimization == 'riemannian':
                G = gradient(X1,X2,V)
                riemannian_grad = .5 * (G - V @ (G.T @ V))
                V += lr_pca * riemannian_grad
                V, _ = np.linalg.qr(V)
            elif pca_optimization == 'landing': 
                G = gradient(X1,X2,V)
                riemannian_grad = .5 * (G - V @ (G.T @ V))
                field = V @ (V.T @ V) - V
                V += lr_pca * (riemannian_grad + landing_lambda * field)
            elif pca_optimization == 'history':
                for m2 in range(1,n_iter_history+1): 
                    S = (sum(n_local)/sum(n_global)) * gradient(X1,X2,V)  \
                        + ((sum(n_global) - sum(n_local))/sum(n_global)) * _V @ D_V @ _V.T @ V
                    V, _ = np.linalg.qr(S)
                D_V = np.diag(np.linalg.norm(S, ord=2, axis=0))
            
            self.evaluate(R=R, V=V)
            if tol is not None:
                _gr = grassmanian_distance(V, _V)
                _nrm = np.sqrt(np.linalg.norm(V.T @ self.Sigma_gap @ V, 2)**2 + np.linalg.norm(self.mu_gap.T @ V)**2)  # fairness constraint
                desc = f"gap={_gr:.6f}, norm_sum={_nrm:.6f}"
                if subspace_optimization == 'history' and constraint in ['covariance', 'all']:
                    desc += f", InnerBrk={m}/{n_iter_history}"
                if pca_optimization == 'history':
                    desc += f", InnerBrk2={m2}/{n_iter_history}"
                pbar.set_description(desc)
                if _gr < tol:
                    print(f"OUTER: Broke in {t} / {n_iter}"); break
        return V


    def transform(self, x:np.ndarray, V=None) -> np.ndarray:
        if V is None:
            assert hasattr(self, 'V'), 'Training is NOT done'
            V = self.V
        return np.dot(V, np.dot(V, x))
    
    
    def plot_buffer(self, n_iter=None, show=False, save=None, fig=None, axes=None):
        plt.close('all')
        if n_iter is None: n_iter = self.n_iter
        if fig is None or axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(9, 6))

        ax = axes[0][0]
        ax.plot(self.buffer.explained_variance_ratio)
        if hasattr(self, 'exp_var_gt') and self.exp_var_gt is not None:
            ax.hlines(self.exp_var_gt/self.total_var, 0, n_iter+1, colors='r', linestyles='dashed')
        ax.hlines(self.eigval_Sigma[-self.k:].sum()/self.total_var, 0, n_iter+1, colors='b', linestyles='dashed')
        ax.set_title('Explained Variance Ratio')
        # ax.set_ylim(top=1)

        ax = axes[0][1]
        ax.plot(self.buffer.objective_unfairness)
        ax.set_title('Objective unfairness')
        ax.set_ylim(bottom=0)

        ax = axes[0][2]
        ax.plot(self.buffer.distance_from_gt)
        ax.set_title('Grassmanian dist. to $V^\\ast$')
        ax.set_ylim(bottom=0)
        if len(self.buffer.distance_from_gt) > 0:
            ax.set_ylim(top=max(self.buffer.distance_from_gt)+.01)

        ax = axes[1][2]
        ax.plot(self.buffer.R_dist)
        ax.set_title('$|| RR^T - R_{opt}R_{opt}^T ||$')
        ax.set_ylim(bottom=0)
        if len(self.buffer.R_dist) > 0:
            ax.set_ylim(top=max(self.buffer.R_dist)+.01)

        ax = axes[1][0]
        ax.plot(self.buffer.projected_mean_diff)
        ax.set_title('Projected mean gap\n(mean fairness)')
        ax.set_ylim(bottom=0)

        ax = axes[1][1]
        ax.plot(self.buffer.projected_cov_diff)
        ax.set_title('Projected cov. gap\n(Cov. fairness)')
        ax.set_ylim(bottom=0)

        # ax = axes[1][2]
        # ax.plot(self.buffer.projected_R_norm)
        # ax.set_title("$tr(R^T \Sigma_{gap} R)$")
        # ax.set_ylim(bottom=0)

        fig.tight_layout()
        if save is not None:
            fig.savefig(save)
        if show:
            fig.show()
        
        return fig, axes
    

def get_args():
    parser = ArgumentParser()
    
    ## Data Generation
    parser.add_argument('-d', '--data_dim', type=int, default=100)
    parser.add_argument('-p', '--probability', type=float, default=0.5)
    parser.add_argument('-R', '--nullity', type=int, default=10)
    parser.add_argument('--seed_data', type=int, default=None)
    parser.add_argument('--mu_scale', type=float, default=1.)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--max_cov_eigs', nargs=2, type=float, default=[1., 2.])

    ## PCA
    parser.add_argument('-k', '--target_dim', type=int, default=3)
    parser.add_argument('-r', '--rank', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--constraint', type=str, default='all')
    parser.add_argument('--mean_estimation', type=str, default='local')     # ['local', 'global']
    parser.add_argument('--subspace_optimization', type=str, default='oja')  # ['oja', 'history']
    parser.add_argument('--pca_optimization', type=str, default='oja')      # ['oja', 'npm', 'history']

    parser.add_argument('--lr_subspace', type=float, default=0.1)           # must be float when pca_optimization in ['oja']
    parser.add_argument('--lr_pca', type=float, default=0.1)                # must be int when subspace_optimization in ['history'] or pca_optimization in ['history']
    parser.add_argument('--landing_lambda', type=float, default=1)        # must be float when subspace_optimization == 'landing' 
    parser.add_argument('--n_iter_history', type=int, default=1)
    parser.add_argument('--seed_train', type=int, default=None)
    parser.add_argument('--tolerance', type=float, default=None)
    args = parser.parse_args()
    return args


def run(args):
    Algo = StreamingFairBlockPCA(
        data_dim=args.data_dim,
        num_attributes=1,
        num_groups=2,
        probability=args.probability,
        nullity=args.nullity,
        seed=args.seed_data,
        mu_scale=args.mu_scale,
        eps=args.eps,
        max_cov_eig0=args.max_cov_eigs[0],
        max_cov_eig1=args.max_cov_eigs[1],
    )

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    w0, _ = np.linalg.eigh(Algo.Sigma0)
    w1, _ = np.linalg.eigh(Algo.Sigma1)
    w, _ = np.linalg.eigh(Algo.Sigma)
    wg, _ = np.linalg.eigh(Algo.Sigma_gap)

    ax[0].plot(w0, label='Eigenvalues: Sigma0')
    ax[0].plot(w1, label='Eigenvalues: Sigma1')
    ax[0].legend()
    ax[1].plot(w, label='Eigenvalues: Sigma')
    ax[1].plot(wg,label='Eigenvalues: Sigma_gap')
    ax[1].legend()
    ax[2].plot(Algo.mu_gap, c='tab:green', label='Entries: $\mu_{gap}$=$\mu_1$-$\mu_0$')
    ax[2].plot(Algo.mu0, c='tab:orange', label='Entries: $\mu_0$')
    ax[2].plot(Algo.mu1, c='tab:red', label='Entries: $\mu_1$')
    ax[2].plot(Algo.mu, c='gray', label='Entries: $\mu=(1-p)\mu_0 + p\mu_1$')
    ax[2].legend()
    # fig.show()
    fig.savefig('records/eigen_info.pdf', bbox_inches='tight')

    Algo.train(
        target_dim=args.target_dim,
        rank=args.rank,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        constraint=args.constraint,
        mean_estimation=args.mean_estimation,
        subspace_optimization=args.subspace_optimization,
        pca_optimization=args.pca_optimization,
        lr_subspace=args.lr_subspace, 
        lr_pca=args.lr_pca,
        n_iter_history=args.n_iter_history,
        landing_lambda=args.landing_lambda,
        seed=args.seed_train,
        tol=args.tolerance,
        # lr_scheduler=None,
        # subspace_only=args.subspace_only,
    )

    Algo.plot_buffer(n_iter=args.n_iter, save='records/run_info.pdf')


if __name__ == "__main__":
    run(get_args())
    