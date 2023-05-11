from argparse import ArgumentParser
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
import numpy as np
from scipy.linalg import null_space
from typing import Iterable, Tuple
from tqdm.auto import trange
# from sklearn.covariance import shrunk_covariance, ledoit_wolf, oas


def sin(A:jnp.ndarray, B:jnp.ndarray):
    assert A.shape == B.shape
    denom = jnp.sqrt(jnp.square(A).sum() * jnp.square(B).sum())
    cos = jnp.sum(A*B) / (denom + 1e-8)
    return jnp.sqrt(1-jnp.square(cos))


def grassmanian_distance(A:jnp.ndarray, B:jnp.ndarray):
    """
    distance between matrix A and B with same size,
    invariant to orthogonal transformation of basis
    """
    assert A.shape == B.shape
    assert A.ndim <= 2
    if A.ndim == 1:
        return jnp.linalg.norm(jnp.outer(A,A)-jnp.outer(B,B))
    assert A.shape[0] >= A.shape[1]
    return jnp.linalg.norm(A @ A.T - B @ B.T)


def truncate_rows(A:jnp.ndarray, leave=None):
    assert A.ndim == 2, A.ndim
    d, k = A.shape
    if leave is None: leave = k
    if d > k:
        norms = jnp.linalg.norm(A, ord=2, axis=0)
        # A[np.argsort(norms)[:d-leave]] = 0
        A = A.at[jnp.argsort(norms)[:d-leave]].set(0)
    return A

    
class Buffer:
    """
    Arbitrary Memory buffer to record performance of PCA algorithm.
    """
    def __init__(self):
        self.explained_variance_ratio = []   # objective function, [0, 1]
        self.distance_from_gt = []           # Grassmanian distance of V and V_ground_truth
        self.R_dist = []                     # Grassmanian distance of R and R_opt
        self.projected_mean_diff = []        # intensity of mean equalization
        self.projected_cov_diff = []         # intensity of covariance equalization
        self.mu_gap_estim_err = []           # cosine between mu_gap and f

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
            rank=None,
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
        if rank is not None:
            assert isinstance(rank, int) and 0 <= rank <= data_dim, f"Invalid rank: {rank}"

        self.d = data_dim
        self.num_attr = num_attributes
        self.num_groups = num_groups
        self.p = probability
        self._r = rank if rank is not None else (data_dim // 2)

        ## Random generator
        # rng = np.random.default_rng(seed) 
        if seed is None:
            from sys import maxsize
            seed = np.random.randint(0,maxsize)
        key = random.PRNGKey(seed)

        ## Conditional mean vectors:
        ## (1-p) * mu0 + p * mu1 = 0  (Assuming true mean == zero vector)
        # mu_temp = rng.standard_normal(self.d)
        key, rng = random.split(key)
        mu_temp = random.normal(rng, (self.d,))
        mu_temp *= mu_scale / jnp.linalg.norm(mu_temp) 
        self.mu0 = mu_temp * self.p / (self.p - 1.)
        self.mu1 = mu_temp
        self.mu = (1-self.p) * self.mu0 + self.p * self.mu1
        self.mu_gap = self.mu1 - self.mu0   # f
        assert jnp.isclose(jnp.abs(self.mu).max(), 0.), f"mu is not close to 0."
        
        ## Conditional covariances:
        ### 1. Orthogonal matrices W0 & W1 (sharing r same columns)
        dim_gap_ = self.d-(self._r//2)
        key, rng = random.split(key)
        A_ = random.normal(rng, (self.d, self._r//2))
        key, rng = random.split(key)
        A0 = random.normal(rng, (self.d, dim_gap_))
        A0 = jnp.concatenate([A_, A0], 1)
        key, rng = random.split(key)
        A1 = random.normal(rng, (self.d, dim_gap_))
        A1 = jnp.concatenate([A_, A1], 1)
        W0, _ = jnp.linalg.qr(A0)   # orthogonal
        W1, _ = jnp.linalg.qr(A1)   # orthogonal
        
        ### 2. Eigenvalues D0 & D1 (sharing r same eigenvalues)
        dim_gap = self.d-self._r
        key, rng = random.split(key)
        D0 = max_cov_eig0 * (1 + 0.1 * random.normal(rng, (self._r,)))  # main
        key, rng = random.split(key)
        D0 = jnp.concatenate([D0, max_cov_eig0 * (jnp.arange(2,dim_gap+2) ** (-1.) + 0.1 * random.normal(rng, (dim_gap,)))])  # power law decay
        D0 = jnp.abs(D0)
        key, rng = random.split(key)
        D1 = max_cov_eig1 * (1 + 0.1 * random.normal(rng, (self._r,)))  # main
        key, rng = random.split(key)
        D1 = jnp.concatenate([D1, max_cov_eig1 * (jnp.arange(2,dim_gap+2) ** (-1.) + 0.1 * random.normal(rng, (dim_gap,)))])  # power law decay
        D1 = jnp.abs(D1)

        ### 3. Eigen-Composition to make Sigma0 & Sigma1;
        ###     rank(Sigma1 - Sigma0) <= d - r.
        self.Sigma0 = W0 @ jnp.diag(D0) @ W0.T
        self.Sigma1 = W1 @ jnp.diag(D1) @ W1.T
        self.Sigma = (1-self.p) * self.Sigma0 + self.p * self.Sigma1 + self.p*(1-self.p) * jnp.outer(self.mu_gap, self.mu_gap)
        assert jnp.min(jnp.linalg.eigh(self.Sigma)[0]) >= 0, f"Invalid Sigma: non-PSD, {jnp.linalg.eigh(self.Sigma)[0]}"
        self.Sigma_gap = self.Sigma1 - self.Sigma0
        
        ## Info about gaps
        # rank_Sigma_gap = np.linalg.matrix_rank(self.Sigma_gap, hermitian=True)
        self.trace_Sigma_gap = jnp.trace(self.Sigma_gap)
        self.Sigma_gap_sq = self.Sigma_gap @ self.Sigma_gap
        self.eigval_Sigma_gap_sq, self.eigvec_Sigma_gap_sq = jnp.linalg.eigh(self.Sigma_gap_sq)  # eigenvalue of Q^2 increasing order
        # assert rank_Sigma_gap == dim_gap, f"Invalid rank(Sigma_gap)={rank_Sigma_gap} (> {dim_gap}=d-r)"  # This command checks rank validity 
        
        ## Optima / group-conditional optima
        self.eigval_Sigma, self.eigvec_Sigma = jnp.linalg.eigh(self.Sigma)  # eigenvalues in ascending order
        # self.total_var, self.V_star = self.eigval_Sigma.sum(), self.eigvec_Sigma
        self.group_vars, self.group_V_stars = [0]*num_groups, [0]*num_groups
        for s in range(self.num_groups):
            eigenvalues, eigenvectors = jnp.linalg.eigh(eval(f"self.Sigma{s}"))
            self.group_vars[s] = jnp.sum(eigenvalues)
            self.group_V_stars[s] = eigenvectors
        _, self.eigvec_Q_hat,= jnp.linalg.eigh(self.Sigma_gap + (jnp.outer(self.mu1,self.mu1) - jnp.outer(self.mu0, self.mu0)))
    

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
            - groud_truth (jnp.ndarray) : size=(data_dim, target_dim)
        """
        assert isinstance(target_dim, int) and 0 < target_dim < self.d, f"Invalid target_dim: {target_dim}"
        g = self.mu_gap 
        norm_g = jnp.linalg.norm(self.mu_gap)
        if not jnp.isclose(norm_g, 0):
            g /= norm_g

        if mode == 'mean':
            R = null_space(g.reshape(1,-1))
        elif mode == 'covariance':
            R = self.eigvec_Sigma_gap_sq[:,:self.d-rank]
        elif mode == 'all': 
            N = self.eigvec_Sigma_gap_sq[:,-rank:]
            N = jnp.concatenate([g.reshape(-1,1), N], 1)
            N, _ = jnp.linalg.qr(N)
            RR = jnp.eye(self.d) - N @ N.T
            D, Q = jnp.linalg.eigh(RR)
            R = Q @ jnp.diag(jnp.sqrt(jnp.abs(D)))
        else:
            return self.eigval_Sigma[-target_dim:].sum(), self.eigvec_Sigma[:,-target_dim:]
        M = R.T @ self.Sigma @ R
        eigval, eigvec = jnp.linalg.eigh(M)
        explained_variance = jnp.sum(eigval[-target_dim:])
        ground_truth = R @ eigvec[:,-target_dim:]
        # if mode in ['mean', 'all']:
        #     assert np.max(np.abs(ground_truth.T @ self.mu_gap)) < 1e-6, \
        #         f"Failed to equalizing means, {np.max(np.abs(ground_truth.T @ self.mu_gap))}"
        return explained_variance, ground_truth
    
    
    def get_explained_variance_ratio(self, V:jnp.ndarray=None, VVT:jnp.ndarray=None):
        """
        Given V, get explained_variance_ratio (== total objective function)

        :: Input
            - V (jnp.ndarray) : size=(data_dim, target_dim)

        :: Return
            - exp_var (float)
        """
        assert hasattr(self, 'total_var')
        if VVT is not None:
            return jnp.trace(self.Sigma @ VVT) / self.total_var
        else:
            return jnp.trace(self.Sigma @ V @ V.T) / self.total_var


    def get_group_explained_variance_ratio(self, s, V:jnp.ndarray=None, VVT:jnp.ndarray=None):
        """
        Given V, get group-conditional explained_variance

        :: Input
            - V (jnp.ndarray) : size=(data_dim, target_dim)

        :: Return
            - exp_var (float)
        """
        assert isinstance(s, int) and 0 <= s < self.num_groups
        assert hasattr(self, 'group_vars')
        if VVT is not None:
            return jnp.trace(eval(f"self.Sigma{s}") @ VVT) / self.group_vars[s]
        else:
            return jnp.trace(eval(f"self.Sigma{s}") @ V @ V.T) / self.group_vars[s]
    

    def get_objective_unfairness(self, V:jnp.ndarray=None, VVT:jnp.ndarray=None):
        """
        Given V, get objective unfairness (difference in conditional objective function)

        :: Input
            - V (jnp.ndarray) : size=(data_dim, target_dim)

        :: Return
            - exp_var (float)
        """
        exp_var_ratio = self.get_explained_variance_ratio(V=V, VVT=VVT)
        exp_var_group_ratios = [self.get_group_explained_variance_ratio(s, V=V, VVT=VVT) for s in range(self.num_groups)]
        return max(abs(exp_var_ratio - exp_var_group_ratios[s]) for s in range(self.num_groups))


    def evaluate(self, V:jnp.ndarray=None, R:jnp.ndarray=None, f:jnp.ndarray=None, rank=None, mode='vanilla'):
        if V is not None:
            VVT = V @ V.T
            k = V.shape[1]
            if not hasattr(self, 'exp_var_gt'):
                if rank is None and hasattr(self, 'r'): rank = self.r
                self.exp_var_gt, self.V_ground_truth = self.get_ground_truth(k, rank, mode=mode)
            self.buffer.explained_variance_ratio.append(self.get_explained_variance_ratio(VVT=VVT))
            if self.V_ground_truth is not None:
                self.buffer.distance_from_gt.append(jnp.linalg.norm(self.V_ground_truth @ self.V_ground_truth.T - VVT))
            self.buffer.projected_mean_diff.append(jnp.linalg.norm(self.mu_gap.T @ V))
            self.buffer.projected_cov_diff.append(jnp.linalg.norm(V.T @ self.Sigma_gap @ V))
        if R is not None:
            r = R.shape[1]
            R_true = self.eigvec_Sigma_gap_sq[:,-r:]
            # R_true = self.eigvec_Q_hat[:,:r]
            self.buffer.R_dist.append(grassmanian_distance(R, R_true))
        if f is not None:
            self.buffer.mu_gap_estim_err.append(sin(f, self.mu_gap))


    def offline_train(self,
            target_dim,
            rank,
            n_iter,
            lr,
            seed=None,
            constraint='all',   # ['vanilla', 'mean', 'covariance', 'all']
            subspace_optimization='pm',    
            pca_optimization='pm',
            n_iter_inner=None,
            lr_scheduler=None,
            landing_lambda = .1,
            fairness_tradeoff=1.,
        ):
        """
        train V in offline manner with fairness constraint
        """
        ## Check arguments
        assert isinstance(target_dim, int) and target_dim < self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert isinstance(lr, (int, float)) and lr > 0
        if constraint != 'vanilla':
            assert subspace_optimization in ['pm']
            assert isinstance(n_iter_inner, int) and n_iter_inner > 0
        assert pca_optimization in ['oja', 'pm', 'riemannian', 'landing']
        assert constraint in ['vanilla', 'mean', 'covariance', 'all']

        # rng = np.random.default_rng(seed)
        if seed is None:
            from sys import maxsize
            seed = np.random.randint(0,maxsize)
        key = random.PRNGKey(seed)
        self.n_iter = n_iter
        self.k = target_dim
        if constraint in ['covariance', 'all']:
            self.r = rank
        
        ## Buffers
        self.buffer = Buffer()
        self.total_var, self.V_star = self.get_ground_truth(self.k, rank, mode=None)
        self.exp_var_gt, self.V_ground_truth = None, None
        if constraint in ['mean', 'covariance', 'all']:
            self.exp_var_gt, self.V_ground_truth = self.get_ground_truth(self.k, rank, mode=constraint)

        R = None
        if constraint in ['mean', 'all']:
            ## Mean gap
            f = self.mu_gap / (jnp.linalg.norm(self.mu_gap) + 1e-8)
            N = f.reshape(-1, 1)   # this line is used only when `constrain == 'mean'`
        if constraint in ['covariance','all']:
            ## Covariance gap
            key, rng = random.split(key)
            R, _ = jnp.linalg.qr(random.normal(rng, (self.d, self.r)))
            N = R.copy()   # this line is used only when `constrain == 'covariance'`

        # V, _ = np.linalg.qr(rng.standard_normal((self.d, self.k)))
        key, rng = random.split(key)
        V, _ = jnp.linalg.qr(random.normal(rng, (self.d, self.k)))
        self.evaluate(V=V, R=R)

        if constraint in ['mena', 'covariance', 'all']:
            for _ in trange(1, n_iter_inner+1):
                # G = self.Sigma_gap @ R
                G = (self.Sigma_gap + (jnp.outer(self.mu1,self.mu1) - jnp.outer(self.mu0, self.mu0))) @ R
                if subspace_optimization == 'pm': R = G
                R, _ = jnp.linalg.qr(R)
                self.evaluate(R=R)
        
        if constraint == 'all':
            ## Normal subspace; for both mean & covariance gap
            if jnp.isclose(jnp.linalg.norm(f), 0):
                N = R.copy()
            else:
                N = jnp.concatenate([f.reshape(-1,1), R], 1)
                N, _ = jnp.linalg.qr(N)
        
        lr0 = lr
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _V = V.copy()  # store previous V
            if lr_scheduler is not None:
                lr = lr0 * lr_scheduler(t)
            ## Gradient
            if constraint == 'vanilla': G = self.Sigma @ V
            else: 
                G = self.Sigma @ (V - fairness_tradeoff * N @ N.T @ V)
                G -= fairness_tradeoff * N @ N.T @ G
            ## Update
            if pca_optimization == 'oja':   V += lr * G
            elif pca_optimization == 'pm':  V = G
            elif pca_optimization == 'riemannian': 
                riemannian_grad = (G - V @ (G.T @ V))
                V += lr * riemannian_grad
            elif pca_optimization == 'landing':
                riemannian_grad = (G - V @ (G.T @ V))
                field = V @ (V.T @ V) - V
                V += lr * (riemannian_grad + landing_lambda * field)
            if pca_optimization != 'landing':
                V, _ = jnp.linalg.qr(V)
            
            self.evaluate(V=V)
        return V
    

    def sample(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Receive a pair of data `x` and corresponding sensitive attribute `s`

        :: Input
            - key (jax.numpy.PRNGKeyArray, optional)

        :: Return
            - s (int) : {0, 1}
            - x (jnp.ndarray) : size=(data_dim, 1)
        """
        if hasattr(self, 'key'):
            # s = rng.binomial(1, self.p)
            # x = rng.multivariate_normal(eval(f"self.mu{s}"), eval(f"self.Sigma{s}"))
            self.key, rng = random.split(self.key)
            s = random.bernoulli(rng, self.p).astype(int)
            self.key, rng = random.split(self.key)
            x = random.multivariate_normal(rng, mean=eval(f"self.mu{s}"), cov=eval(f"self.Sigma{s}"))
        else:
            s = np.random.binomial(1, self.p)
            x = np.random.multivariate_normal(mean=eval(f"self.mu{s}"), cov=eval(f"self.Sigma{s}"))
        return s, x
    

    def subspace_estimation(self,
            R,
            t,
            rank,
            batch_size,
            n_global_group,
            mean_global_group,
            constraint,
            subspace_optimization,
            subspace_frequent_direction,
            B_R=None,
            D_R=None
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Iterable, Iterable]:
        """
        Orthogonal subspace estimation for Fair Streaming PCA.
        """

        f = None
        if R is not None: _R = R.copy()       # store previous R

        ## Before Sampling
        n_local_group = [0 for _ in range(self.num_groups)]  # n per group, local
        if constraint in ['mean', 'all']:
            mean_local_group = [jnp.zeros(self.d) for _ in range(self.num_groups)]
        if constraint in ['covariance', 'all']:
            cov_R_group = [jnp.zeros((self.d, rank)) for _ in range(self.num_groups)]

        ## Sampling
        for _ in range(batch_size):
            s, x = self.sample()
            if constraint in ['mean', 'all']:
                mean_local_group[s] += x
            if constraint in ['covariance', 'all']:
                cov_R_group[s] += jnp.outer(x, jnp.dot(x, R))
            n_local_group[s] += 1
        
        ## After Sampling
        for s in range(self.num_groups):
            mean_local_group[s] /= batch_size
            cov_R_group[s] /= batch_size
            if constraint in ['mean', 'all']:
                mean_global_group[s] *= n_global_group[s]
                mean_global_group[s] += n_local_group[s] * mean_local_group[s]
                mean_global_group[s] /= n_global_group[s] + n_local_group[s]
            n_global_group[s] += n_local_group[s]

        if constraint in ['covariance', 'all']:
            ## Covariance gap
            covdiff_R = (n_local_group[0] * cov_R_group[1] - n_local_group[1] * cov_R_group[0]) / batch_size
            if subspace_optimization == 'npm': 
                R = covdiff_R
                R, _ = jnp.linalg.qr(R)
            elif subspace_optimization == 'history':
                if t==1:
                    S = R + covdiff_R
                else:
                    S = (n_local_group[s]/n_global_group[s]) * covdiff_R  \
                        + ((n_global_group[s] - n_local_group[s])/n_global_group[s]) * _R @ jnp.diag(D_R) @ _R.T @ R
                R, _ = jnp.linalg.qr(S)
                D_R = jnp.linalg.norm(S, ord=2, axis=0)
            if subspace_frequent_direction:
                B_R = B_R.at[:,-self.r:].set(R)
                U_R, S_R, _ = jnp.linalg.svd(B_R, full_matrices=False)  # singular value in decreasing order
                min_singular = jnp.square(S_R[self.r])
                S_R = jnp.sqrt(jnp.clip(jnp.square(S_R)-min_singular, 0, None))
                B_R = U_R @ jnp.diag(S_R)
                R = B_R[:,:self.r]
                R, _ = jnp.linalg.qr(R)
        
        return R, n_global_group, mean_global_group


    def train(self,
            target_dim,
            rank,
            n_iter,
            batch_size_subspace,
            batch_size_pca,
            constraint='all',
            subspace_optimization='npm',
            pca_optimization='npm',
            n_iter_inner=None,
            lr_pca=None,
            landing_lambda=None,
            seed=None,
            verbose=True,
            lr_scheduler=None,
            fairness_tradeoff=1.,
        ):
        """
        Streaminig Fair Block PCA Algorithm.

        """
        ## Check arguments
        assert isinstance(target_dim, int) and 0 < target_dim < self.d
        assert isinstance(rank, int) and 0 < rank <= self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert constraint in ['vanilla', 'mean', 'covariance', 'all'], constraint
        subspace_frequent_direction = False
        if constraint != 'vanilla':
            assert subspace_optimization in ['npm', 'npmfd', 'history', 'historyfd'], subspace_optimization
            assert isinstance(batch_size_subspace, int) and batch_size_subspace > 0
            if subspace_optimization[-2:] == 'fd':
                subspace_optimization = subspace_optimization[:-2]
                subspace_frequent_direction = True
        pca_frequent_direction = False
        assert pca_optimization in ['oja', 'npm', 'ojafd', 'npmfd', 'riemannian', 'landing', 'history', 'historyfd'], pca_optimization
        assert isinstance(batch_size_pca, int) and batch_size_pca > 0
        if pca_optimization[-2:] == 'fd':
            pca_optimization = pca_optimization[:-2]
            pca_frequent_direction = True
        if pca_optimization not in ['npm', 'npmfd', 'history']:
            assert isinstance(lr_pca, (int, float)) and lr_pca > 0
        if pca_optimization in ['landing']:
            assert isinstance(landing_lambda, float) and landing_lambda > 0

        ## Randomness
        if seed is None:
            from sys import maxsize
            seed = np.random.randint(0,maxsize)
        self.key = random.PRNGKey(seed)
        self.n_iter = n_iter
        self.k = target_dim
        if constraint in ['covariance', 'all']:
            self.r = rank

        ## Buffers
        self.buffer = Buffer()

        ## Optimality
        self.total_var, self.V_star = self.get_ground_truth(self.k, rank, mode=None)
        self.exp_var_gt, self.V_ground_truth = None, None
        if constraint in ['mean', 'covariance', 'all']:
            self.exp_var_gt, self.V_ground_truth = self.get_ground_truth(self.k, rank, mode=constraint)
        
        ## Optimization variables
        f, R = None, None
        if constraint in ['covariance', 'all']:
            self.key, rng = random.split(self.key)
            R, _ = jnp.linalg.qr(random.normal(rng, (self.d, rank)))
        self.key, rng = random.split(self.key)
        V, _ = jnp.linalg.qr(random.normal(rng, (self.d, self.k)))
        self.evaluate(V=V, R=R)

        ## SUBSPACE ESTIMATION
        if constraint in ['mean', 'covariance', 'all']:
            n_global_group = [0 for _ in range(self.num_groups)]  # n per group
            mean_global_group = [jnp.zeros(self.d) for _ in range(self.num_groups)]
            B_R, D_R = None, None
            if subspace_frequent_direction: B_R = jnp.zeros((self.d, 2*rank))
            if subspace_optimization == 'history': D_R = jnp.zeros(rank)

            pbar = trange(1, n_iter_inner+1)
            for t in pbar:
                R, n_global_group, mean_global_group = self.subspace_estimation(
                    R, t, self.r, batch_size_subspace, n_global_group, mean_global_group,
                    constraint, subspace_optimization, subspace_frequent_direction, B_R, D_R
                )
                self.evaluate(R=R)
                if verbose:
                    desc = ""
                    if constraint in ['covarinace', 'all']:
                        desc += f"R_dist={self.buffer.R_dist[-1]:.5f} "
                    pbar.set_description(desc)

            if constraint in ['mean', 'all']:
                ## Mean gap
                assert n_iter_inner * batch_size_subspace == sum(n_global_group), (n_iter_inner * batch_size_subspace, sum(n_global_group))
                f = (n_global_group[0] * mean_global_group[1] - n_global_group[1] * mean_global_group[0]) / (n_iter_inner * batch_size_subspace-1)
                f /= (jnp.linalg.norm(f) + 1e-8)
                self.evaluate(f=f)
                N = f.reshape(-1, 1)   # this line is used only when `constrain == 'mean'`

            if constraint in ['covariance', 'all']:
                N = R.copy()   # this line is used only when `constrain == 'covariance'`
                            
            if constraint == 'all':
                ## Normal subspace; for both mean & covariance gap
                if jnp.isclose(jnp.linalg.norm(f), 0):
                    N = R.copy()
                else:
                    N = jnp.concatenate([f.reshape(-1,1), R], 1)
                    N, _ = jnp.linalg.qr(N)
        
        ## PCA OPTIMIZATION
        n_global = 0
        mean_global = jnp.zeros(self.d)
        B_V, D_V = None, None
        if pca_frequent_direction: B_V = jnp.zeros((self.d, 2*self.k))
        if pca_optimization == 'history': D_V = jnp.zeros(self.k)
        lr_pca0 = lr_pca
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _V = V.copy()       # store previous V
            if lr_scheduler is not None and lr_pca is not None: lr_pca = lr_pca0 * lr_scheduler(t)

            ## Before Sampling
            mean_local = jnp.zeros(self.d)
            cov_V =  jnp.zeros((self.d,self.k))
            if constraint in ['mean', 'covariance', 'all']:
                ## projection
                V -= fairness_tradeoff * N @ N.T @ V

            ## Sampling
            for n_local in range(batch_size_pca):
                _, x = self.sample()
                # mean_local *= n_local
                mean_local += x / batch_size_pca
                # mean_local /= n_local+1
                # cov_V *= n_local
                cov_V += jnp.outer(x, jnp.dot(x, V)) / batch_size_pca
                # cov_V /= n_local+1

            ## After Sampling
            # cov_V -= np.outer(mean_global, np.dot(mean_global, V))
            mean_global *= n_global
            mean_global += batch_size_pca * mean_local
            mean_global /= n_global + n_local
            n_global += n_local
            if constraint in ['mean', 'covariance', 'all']:
                ## projection
                cov_V -= fairness_tradeoff * N @ N.T @ cov_V
            
            if pca_optimization == 'oja':
                V += lr_pca * cov_V
                V, _ = jnp.linalg.qr(V)
            elif pca_optimization == 'npm':
                V = cov_V
                V, _ = jnp.linalg.qr(V)
            elif pca_optimization == 'riemannian':
                G = cov_V
                riemannian_grad = (G - V @ (G.T @ V))
                V += lr_pca * riemannian_grad
                V, _ = jnp.linalg.qr(V)
            elif pca_optimization == 'landing': 
                G = cov_V
                riemannian_grad = (G - V @ (G.T @ V))
                field = V @ (V.T @ V) - V
                V += lr_pca * (riemannian_grad + landing_lambda * field)
            elif pca_optimization == 'history':
                if t==1:
                    S = V + cov_V
                else:
                    S = batch_size_pca * cov_V + (n_global - batch_size_pca) * _V @ jnp.diag(D_V) @ _V.T @ V
                    S /= n_global
                V, _ = jnp.linalg.qr(S)
                D_V = jnp.linalg.norm(S, ord=2, axis=0)
            if pca_frequent_direction:
                B_V = B_V.at[:,-self.k:].set(V)
                U_V, S_V, _ = jnp.linalg.svd(B_V, full_matrices=False)  # singular value in decreasing order
                min_singular = jnp.square(S_V[self.k])
                S_V = jnp.sqrt(jnp.clip(jnp.square(S_V)-min_singular, 0, None))
                B_V = U_V @ jnp.diag(S_V)
                V = B_V[:,:self.k]
                V, _ = jnp.linalg.qr(V)
            
            self.evaluate(V=V)
            if verbose:
                desc = "" 
                if constraint in ['mean', 'all']:
                    desc += f"mu_gap_err={self.buffer.mu_gap_estim_err[-1]:.5f} "
                desc += f"mu_err={jnp.linalg.norm(mean_global-self.mu):.5f} "
                pbar.set_description(desc)
            

        return V
    

    def train_combined(self,
            target_dim,
            rank,
            n_iter,
            batch_size_subspace,
            batch_size_pca,
            constraint='all',
            subspace_optimization='npm',
            pca_optimization='oja',
            lr_pca=None,
            n_iter_inner=1,
            landing_lambda=None,
            seed=None,
            tol=None,
            lr_scheduler=None,
            fairness_tradeoff=1.,
        ):
        """
        Streaminig Fair Block PCA Algorithm.

        """
        ## Check arguments
        assert isinstance(target_dim, int) and 0 < target_dim < self.d
        assert isinstance(rank, int) and 0 < rank <= self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert constraint in ['vanilla', 'mean', 'covariance', 'all'], constraint
        subspace_frequent_direction = False
        if constraint != 'vanilla':
            assert subspace_optimization in ['npm', 'npmfd', 'history', 'historyfd'], subspace_optimization
            assert isinstance(batch_size_subspace, int) and batch_size_subspace > 0
            if subspace_optimization[-2:] == 'fd':
                subspace_optimization = subspace_optimization[:-2]
                subspace_frequent_direction = True
        pca_frequent_direction = False
        assert pca_optimization in ['oja', 'npm', 'ojafd', 'npmfd', 'riemannian', 'landing', 'history', 'historyfd'], pca_optimization
        assert isinstance(batch_size_pca, int) and batch_size_pca > 0
        if pca_optimization[-2:] == 'fd':
            pca_optimization = pca_optimization[:-2]
            pca_frequent_direction = True
        
        if pca_optimization not in ['npm', 'npmfd', 'history']:
            assert isinstance(lr_pca, (int, float)) and lr_pca > 0
        if pca_optimization in ['landing']:
            assert isinstance(landing_lambda, float) and landing_lambda > 0

        if seed is None:
            from sys import maxsize
            seed = np.random.randint(0,maxsize)
        self.key = random.PRNGKey(seed)
        self.n_iter = n_iter
        self.k = target_dim
        if constraint in ['covariance', 'all']:
            self.r = rank

        ## Buffers
        self.buffer = Buffer()
        self.total_var, self.V_star = self.get_ground_truth(self.k, rank, mode=None)
        self.exp_var_gt, self.V_ground_truth = None, None
        if constraint in ['mean', 'covariance', 'all']:
            self.exp_var_gt, self.V_ground_truth = self.get_ground_truth(self.k, rank, mode=constraint)
        
        f, R = None, None
        if constraint in ['covariance', 'all']:
            self.key, rng = random.split(self.key)
            R, _ = jnp.linalg.qr(random.normal(rng, (self.d, rank)))
        self.key, rng = random.split(self.key)
        V, _ = jnp.linalg.qr(random.normal(rng, (self.d, self.k)))
        self.evaluate(V=V, R=R)

        n_global = 0
        mean_global = jnp.zeros(self.d)
        if constraint in ['mean', 'covariance', 'all']:
            n_global_group = [0 for _ in range(self.num_groups)]  # n per group
            if constraint != 'covariance':
                mean_global_group = [jnp.zeros(self.d) for _ in range(self.num_groups)]
        
        B_R, D_R = None, None
        if subspace_frequent_direction: B_R = jnp.zeros((self.d, 2*rank))
        if pca_frequent_direction: B_V = jnp.zeros((self.d, 2*self.k))
        if subspace_optimization == 'history': D_R = jnp.zeros(rank)
        if pca_optimization == 'history': D_V = jnp.zeros(self.k)
        lr_pca0 = lr_pca

        pbar = trange(1, n_iter+1)
        for t in pbar:
            _V = V.copy()       # store previous V
            if lr_scheduler is not None and lr_pca is not None: lr_pca = lr_pca0 * lr_scheduler(t)
            
            ## SUBSPACE ESTIMATION
            if constraint in ['mean', 'covariance', 'all']:
                for _ in range(n_iter_inner):
                    N, f, R, n_global_group, mean_global_group = self.subspace_estimation(
                        R, t, self.r, batch_size_subspace, n_global_group, mean_global_group,
                        constraint, subspace_optimization, subspace_frequent_direction, B_R, D_R
                    )

            ## PCA OPTIMIZATION
            ## Before Sampling
            mean_local = jnp.zeros(self.d)
            cov_V =  jnp.zeros((self.d,self.k))
            if constraint in ['mean', 'covariance', 'all']:
                ## projection
                V -= fairness_tradeoff * N @ N.T @ V

            ## Sampling
            for n_local in range(batch_size_pca):
                _, x = self.sample()
                mean_local *= n_local
                mean_local += x
                mean_local /= n_local+1
                cov_V *= n_local
                cov_V += jnp.outer(x, jnp.dot(x, V))
                cov_V /= n_local+1
                n_local += 1

            ## After Sampling
            # cov_V -= np.outer(mean_global, np.dot(mean_global, V))
            mean_global *= n_global
            mean_global += batch_size_pca * mean_local
            mean_global /= n_global + batch_size_pca
            n_global += n_local

            if constraint in ['mean', 'covariance', 'all']:
                ## projection
                cov_V -= fairness_tradeoff * N @ N.T @ cov_V
            
            if pca_optimization == 'oja':
                V += lr_pca * cov_V
                V, _ = jnp.linalg.qr(V)
            elif pca_optimization == 'npm':
                V = cov_V
                V, _ = jnp.linalg.qr(V)
            elif pca_optimization == 'riemannian':
                G = cov_V
                riemannian_grad = (G - V @ (G.T @ V))
                V += lr_pca * riemannian_grad
                V, _ = jnp.linalg.qr(V)
            elif pca_optimization == 'landing': 
                G = cov_V
                riemannian_grad = (G - V @ (G.T @ V))
                field = V @ (V.T @ V) - V
                V += lr_pca * (riemannian_grad + landing_lambda * field)
            elif pca_optimization == 'history':
                if t==1:
                    S = V + cov_V
                else:
                    S = batch_size_pca * cov_V + (n_global - batch_size_pca) * _V @ jnp.diag(D_V) @ _V.T @ V
                    S /= n_global
                V, _ = jnp.linalg.qr(S)
                D_V = jnp.linalg.norm(S, ord=2, axis=0)
            if pca_frequent_direction:
                B_V = B_V.at[:,-self.k:].set(V)
                U_V, S_V, _ = jnp.linalg.svd(B_V, full_matrices=False)  # singular value in decreasing order
                min_singular = jnp.square(S_V[self.k])
                S_V = jnp.sqrt(jnp.clip(jnp.square(S_V)-min_singular, 0, None))
                B_V = U_V @ jnp.diag(S_V)
                V = B_V[:,:self.k]
                V, _ = jnp.linalg.qr(V)
            
            self.evaluate(V=V, R=R, f=f)
            if tol is not None:
                # _gr = grassmanian_distance(V, _V)
                desc = f"mu_err={jnp.linalg.norm(mean_global-self.mu):.5f} "
                if constraint in ['mean', 'all']:
                    desc += f"mu_gap_err={self.buffer.mu_gap_estim_err[-1]:.5f} "
                if constraint in ['covarinace', 'all']:
                    desc += f"R_dist={self.buffer.R_dist[-1]:.5f} "
                pbar.set_description(desc)
                # if _gr < tol:
                #     print(f"OUTER: Broke in {t} / {n_iter}"); break
        return V


    def transform(self, x:jnp.ndarray, V=None) -> jnp.ndarray:
        if V is None:
            assert hasattr(self, 'V'), 'Training is NOT done'
            V = self.V
        return jnp.dot(V, jnp.dot(V, x))
    

    def plot_buffer(self, n_iter=None, show=False, save=None, fig=None, axes=None):
        plt.close('all')
        if n_iter is None: n_iter = self.n_iter
        if fig is None or axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(9, 6))

        ax = axes[0][0]
        ax.plot(self.buffer.explained_variance_ratio)
        if hasattr(self, 'exp_var_gt') and self.exp_var_gt is not None:
            ax.hlines(self.exp_var_gt/self.total_var, 0, n_iter+1, colors='r', linestyles='dashed')
        ax.hlines(self.eigval_Sigma[-self.k:].sum()/self.total_var, 0, n_iter+1, colors='b', linestyles='dashdot')
        ax.set_title('Explained Variance Ratio')
        # ax.set_ylim(top=1)

        ax = axes[0][1]
        ax.plot(self.buffer.distance_from_gt)
        ax.set_title('Grassmanian dist. to $V^{(k)}_\\ast$')
        ax.set_ylim(bottom=-0.01)
        if len(self.buffer.distance_from_gt) > 0:
            ax.set_ylim(top=max(self.buffer.distance_from_gt)+.01)
        
        ax = axes[0][2]
        ax.plot(self.buffer.mu_gap_estim_err)
        ax.set_title('Estimation error: $\mu_{gap}$')
        ax.set_ylim(bottom=-0.01)
        if len(self.buffer.mu_gap_estim_err) > 0:
            ax.set_ylim(top=max(self.buffer.mu_gap_estim_err)+.01)
        
        ax = axes[1][0]
        ax.plot(self.buffer.projected_mean_diff)
        ax.set_title('Projected mean gap\n(mean fairness)')
        ax.set_ylim(bottom=-0.01)
        # if len(self.buffer.projected_mean_diff) > 0:
        #     ax.set_ylim(top=max(self.buffer.projected_mean_diff)+.01)

        ax = axes[1][1]
        ax.plot(self.buffer.projected_cov_diff)
        ax.set_title('Projected cov. gap\n(Cov. fairness)')
        ax.set_ylim(bottom=-0.01)
        # if len(self.buffer.projected_cov_diff) > 0:
        #     ax.set_ylim(top=max(self.buffer.projected_cov_diff)+.01)

        ax = axes[1][2]
        ax.plot(self.buffer.R_dist)
        ax.set_title('Subspace Estimation:\n$|| RR^T - R_{opt}R_{opt}^T ||_F$')
        ax.set_ylim(bottom=-0.01)
        if len(self.buffer.R_dist) > 0:
            ax.set_ylim(top=max(self.buffer.R_dist)+.01)


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
    parser.add_argument('-R', '--rank_eff', type=int, default=10)
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
    parser.add_argument('--subspace_optimization', type=str, default='oja')  # ['oja', 'history']
    parser.add_argument('--pca_optimization', type=str, default='oja')      # ['oja', 'npm', 'history']

    parser.add_argument('--lr_subspace', type=float, default=0.1)           # must be float when pca_optimization in ['oja']
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
        rank=args.rank_eff,
        seed=args.seed_data,
        mu_scale=args.mu_scale,
        eps=args.eps,
        max_cov_eig0=args.max_cov_eigs[0],
        max_cov_eig1=args.max_cov_eigs[1],
    )

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    w0, _ = jnp.linalg.eigh(Algo.Sigma0)
    w1, _ = jnp.linalg.eigh(Algo.Sigma1)
    w, _ = jnp.linalg.eigh(Algo.Sigma)
    wg, _ = jnp.linalg.eigh(Algo.Sigma_gap)

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
        batch_size_subspace=args.batch_size,
        constraint=args.constraint,
        subspace_optimization=args.subspace_optimization,
        pca_optimization=args.pca_optimization,
        lr_pca=args.lr_pca,
        n_iter_inner=args.n_iter_inner,
        landing_lambda=args.landing_lambda,
        seed=args.seed_train,
        verbose=args.tolerance,
        # lr_scheduler=None,
    )

    Algo.plot_buffer(n_iter=args.n_iter, save='records/run_info.pdf')


if __name__ == "__main__":
    run(get_args())
    