from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
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
        return np.linalg.norm(np.outer(A,A)-np.outer(B,B),2)
    return np.linalg.norm(A @ A.T - B @ B.T,2)

    
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
        rng = np.random.default_rng(seed) 

        ## Conditional mean vectors:
        ## (1-p) * mu0 + p * mu1 = 0  (Assuming true mean == zero vector)
        mu_temp = rng.standard_normal(self.d)
        mu_temp *= mu_scale / np.linalg.norm(mu_temp) 
        self.mu0 = mu_temp * self.p / (self.p - 1.)
        self.mu1 = mu_temp
        self.mu = (1-self.p) * self.mu0 + self.p * self.mu1
        self.mu_gap = self.mu1 - self.mu0   # f
        assert np.isclose(np.abs(self.mu).max(), 0.), f"mu is not close to 0."
        
        ## Conditional covariances:
        ### 1. Orthogonal matrices W0 & W1 (sharing r same columns)
        dim_gap = self.d-self._r
        A_ = rng.standard_normal((self.d, self._r))
        A0 = np.concatenate([A_, rng.standard_normal((self.d, dim_gap))], axis=1)
        A1 = np.concatenate([A_, rng.standard_normal((self.d, dim_gap))], axis=1)
        W0, _ = np.linalg.qr(A0)   # orthogonal
        W1, _ = np.linalg.qr(A1)   # orthogonal
        if self._r > 0:
            assert np.isclose(np.max(W0[:,:self._r] - W1[:,:self._r]), 0.), f"Invalid orthogonal matrices: W0[:,:r] != W1[:,:r]"  # This command also checks rank validity
        
        ### 2. Eigenvalues D0 & D1 (sharing r same eigenvalues)
        D_ = rng.normal(0, max(max_cov_eig0, max_cov_eig1)+1, self._r)
        D0 = np.concatenate([D_, rng.normal(0, max_cov_eig0, dim_gap)])
        D1 = np.concatenate([D_, rng.normal(0, max_cov_eig1, dim_gap)])
        # D0[:self._r] += rng.uniform(-eps, eps, self._r)
        D0, D1 = np.abs(D0) ** 1.5, np.abs(D1) ** 1.5

        ### 3. Eigen-Composition to make Sigma0 & Sigma1;
        ###     rank(Sigma1 - Sigma0) <= d - r.
        self.Sigma0 = W0 @ np.diag(D0) @ W0.T
        self.Sigma1 = W1 @ np.diag(D1) @ W1.T
        self.Sigma = (1-self.p) * self.Sigma0 + self.p * self.Sigma1 + self.p*(1-self.p) * np.outer(self.mu_gap, self.mu_gap)
        assert np.min(np.linalg.eigh(self.Sigma)[0]) >= 0, f"Invalid Sigma: non-PSD, {np.linalg.eigh(self.Sigma)[0]}"
        self.Sigma_gap = self.Sigma1 - self.Sigma0
        
        ## Info about gaps
        rank_Sigma_gap = np.linalg.matrix_rank(self.Sigma_gap, hermitian=True)
        self.trace_Sigma_gap = np.trace(self.Sigma_gap)
        self.Sigma_gap_sq = self.Sigma_gap @ self.Sigma_gap
        self.eigval_Sigma_gap_sq, self.eigvec_Sigma_gap_sq = np.linalg.eigh(self.Sigma_gap_sq)  # eigenvalue of Q^2 increasing order
        # assert rank_Sigma_gap == dim_gap, f"Invalid rank(Sigma_gap)={rank_Sigma_gap} (> {dim_gap}=d-r)"  # This command checks rank validity 
        
        ## Optima
        self.eigval_Sigma, self.eigvec_Sigma = np.linalg.eigh(self.Sigma)  # eigenvalues in ascending order
        self.total_var, self.V_star = self.eigval_Sigma.sum(), self.eigvec_Sigma
        self.group_vars, self.group_V_stars = zip(*[self.get_group_optimum(s) for s in range(self.num_groups)])
    

    def get_group_optimum(self, s):
        """
        Given k (`target_dim`) and attribute index (`s`), 
        get group-conditional optimum of explained variance and optimal PC's

        :: Input
            - s (int)
            - target_dim (int)

        :: Return
            - total_var (float)
            - V_star (np.ndarray) : size=(data_dim, 1)
        """
        assert isinstance(s, int) and 0 <= s < self.num_groups
        eigenvalues, eigenvectors = np.linalg.eigh(eval(f"self.Sigma{s}"))
        return eigenvalues.sum(), eigenvectors


    def get_ground_truth(self, target_dim, rank=None):
        """
        Given k (`target_dim`) and r (`rank`), 
        get groud-truth fair PC matrix.

        :: Input
            - target_dim (int)
            - rank (int, optional)

        :: Return
            - groud_truth (np.ndarray) : size=(data_dim, target_dim)
        """
        assert isinstance(target_dim, int) and 0 < target_dim < self.d, f"Invalid target_dim: {target_dim}"
        if rank is None: rank = target_dim
        else: assert isinstance(rank, int) and target_dim < rank <= self.d, f"Invalid rank: {rank}"

        ## R :: d x r matrix, whose columns are eigenvectors of Sigma_gap corr. to smallest eigenvalues in magnitude
        R = self.eigvec_Sigma_gap_sq[:,:rank]

        ## W :: r x (r-1) matrix : f' * R nullspace
        u = np.dot(self.mu_gap, R)
        W = null_space(u.reshape(1,-1))
        assert np.isclose(np.abs(np.dot(u, W)).max(), 0.), "Invalid nullspace of meam_gap"

        ## M :: (r-1) x (r-1) matrix, which we want to obtain whose eigendecomposition
        M = W.T @ R.T @ self.Sigma @ R @ W
        eigval, Lam = np.linalg.eigh(M)

        explained_variance = np.sum(eigval[-target_dim:])
        ground_truth_Lam = W @ Lam[:,-target_dim:]
        ground_truth = R @ ground_truth_Lam
        assert np.isclose(np.max(np.abs(ground_truth.T @ self.mu_gap)), 0), \
            f"Failed to equalizing means, {np.max(np.abs(ground_truth.T @ self.mu_gap))}"
        # assert np.abs(np.trace(ground_truth.T @ self.Sigma_gap @ ground_truth)) < np.abs(np.sum(self.eigval_Sigma_gap_sq[:rank])), \
        #     f"Failed to equalizing covariances, {np.abs(np.trace(ground_truth.T @ self.Sigma_gap @ ground_truth))} >= {np.abs(np.sum(self.eigval_Sigma_gap_sq[:rank]))}"
        return explained_variance, ground_truth, ground_truth_Lam
    

    def get_V_star(self, target_dim, rank=None):
        """
        Given k (`target_dim`) and r (`rank`), 
        get groud-truth fair PC matrix.

        :: Input
            - target_dim (int)
            - rank (int, optional)

        :: Return
            - groud_truth (np.ndarray) : size=(data_dim, target_dim)
        """
        assert isinstance(target_dim, int) and 0 < target_dim < self.d, f"Invalid target_dim: {target_dim}"
        if rank is None: rank = target_dim
        # else: assert isinstance(rank, int) and target_dim < rank <= self.d, f"Invalid rank: {rank}"

        ## R :: d x (d-1) matrix, whose columns form an orth. basis of nullspace of mu_gap
        R = null_space(self.mu_gap.reshape(1,-1))

        ## Q :: (d-1) x r matrix, smallest eigenvectors of R' * (Sigma_gap^2) * R
        _, Q = np.linalg.eigh(R.T @ self.Sigma_gap @ R)
        Q = Q[:,:rank]

        M = Q.T @ R.T @ self.Sigma @ R @ Q
        eigval, eigvec = np.linalg.eigh(M)

        explained_variance = np.sum(eigval[-target_dim:])
        ground_truth = R @ Q @ eigvec[:,-target_dim:]
        assert np.isclose(np.max(np.abs(ground_truth.T @ self.mu_gap)), 0), \
            f"Failed to equalizing means, {np.max(np.abs(ground_truth.T @ self.mu_gap))}"
        return explained_variance, ground_truth
    
    def get_V_star_v2(self, target_dim, rank=None):
        """
        Given k (`target_dim`) and r (`rank`), 
        get groud-truth fair PC matrix.

        :: Input
            - target_dim (int)
            - rank (int, optional)

        :: Return
            - groud_truth (np.ndarray) : size=(data_dim, target_dim)
        """
        assert isinstance(target_dim, int) and 0 < target_dim < self.d, f"Invalid target_dim: {target_dim}"
        
        if rank is None or rank <= 0:
            R = null_space(self.mu_gap.reshape(1,-1))
        else:
            g = self.mu_gap / np.linalg.norm(self.mu_gap)
            N = self.eigvec_Sigma_gap_sq[:,-rank+1:]
            N = np.concatenate([N, g.reshape(-1,1)], 1)
            N, _ = np.linalg.qr(N)
            RR = np.eye(self.d) - N @ N.T
            D, Q = np.linalg.eigh(RR)
            R = (Q @ np.diag(D))[:,-rank:]

        M = R.T @ self.Sigma @ R
        eigval, eigvec = np.linalg.eigh(M)

        explained_variance = np.sum(eigval[-target_dim:])
        ground_truth = R @ eigvec[:,-target_dim:]
        assert np.isclose(np.max(np.abs(ground_truth.T @ self.mu_gap)), 0), \
            f"Failed to equalizing means, {np.max(np.abs(ground_truth.T @ self.mu_gap))}"
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


    def evaluate(self, R:np.ndarray=None, V:np.ndarray=None):
        if R is not None:
            r = R.shape[1]
            R_true = self.eigvec_Sigma_gap_sq[:,:r]
            self.buffer.R_dist.append(grassmanian_distance(R, R_true))
            self.buffer.projected_R_norm.append(np.linalg.norm(R.T @ self.Sigma_gap @ R))
        if V is not None:
            VVT = V @ V.T
            k = V.shape[1]
            if not hasattr(self, 'total_var'):
                self.total_var, _ = self.get_total_optimum(k)
            if not hasattr(self, 'group_vars'):
                self.group_vars, _ = zip(*[self.get_group_optimum(_s, k) for _s in range(self.num_groups)])
            if not hasattr(self, 'exp_var_gt') or R is None:
                self.exp_var_gt, self.V_ground_truth, self.Lam_ground_truth = self.get_ground_truth(k, r)
            self.buffer.explained_variance_ratio.append(self.get_explained_variance_ratio(VVT=VVT))
            self.buffer.objective_unfairness.append(self.get_objective_unfairness(VVT=VVT))
            self.buffer.distance_from_gt.append(np.linalg.norm(self.V_ground_truth @ self.V_ground_truth.T - VVT,2))
            self.buffer.projected_mean_diff.append(np.linalg.norm(self.mu_gap.T @ V,2))
            self.buffer.projected_cov_diff.append(np.linalg.norm(V.T @ self.Sigma_gap @ V,2))


    def evaluate_reg(self, V:np.ndarray=None, rank=None):
        if V is not None:
            VVT = V @ V.T
            k = V.shape[1]
            if not hasattr(self, 'exp_var_gt'):
                if rank is None: 
                    assert hasattr(self, 'r')
                    rank = self.r
                self.exp_var_gt, self.V_ground_truth = self.get_V_star_v2(k, rank)
            self.buffer.explained_variance_ratio.append(self.get_explained_variance_ratio(VVT=VVT))
            self.buffer.objective_unfairness.append(self.get_objective_unfairness(VVT=VVT))
            self.buffer.distance_from_gt.append(np.linalg.norm(self.V_ground_truth @ self.V_ground_truth.T - VVT,2))
            self.buffer.projected_mean_diff.append(np.linalg.norm(self.mu_gap.T @ V))
            self.buffer.projected_cov_diff.append(np.linalg.norm(V.T @ self.Sigma_gap @ V))
        

    def offline_train(self,
            target_dim,
            rank,
            n_iter,
            lr_R,
            lr_Lam,
            seed=None,
            mode_R='square',
            mode_Lam='oja',  # ['oja', 'pm']
            tol=None,
            lr_scheduler=None,
        ):
        """
        For a fixed (optimal) R, train Lam in offline manner
        """

        ## Check arguments
        assert isinstance(target_dim, int) and target_dim < self.d
        assert isinstance(rank, int) and target_dim < rank <= self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert isinstance(lr_Lam, (int, float)) and lr_Lam > 0
        
        rng = np.random.default_rng(seed)
        self.r = rank  # can be different to self._r
        self.k = target_dim

        ## Buffers
        self.buffer = Buffer()

        Q_inv = np.linalg.inv(np.eye(self.d) + lr_R * self.Sigma_gap)

        ## Optimization starts from here
        Lam, _ = np.linalg.qr(rng.standard_normal((self.r, self.k)))    # Optimization variable
        R, _ = np.linalg.qr(rng.standard_normal((self.d, self.r)))    # Optimization variable
        V = R @ Lam
        self.evaluate(R=R, V=V)

        lr0 = lr_Lam
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _Lam = Lam.copy()  # store previous Lam
            if lr_scheduler is not None:
                lr_Lam = lr0 * lr_scheduler(t)
            if mode_R == 'square':    R, _ = np.linalg.qr(R - lr_R * self.Sigma_gap_sq @ R)
            elif mode_R == 'inverse': R, _ = np.linalg.qr(Q_inv @ R)

            u = np.dot(self.mu_gap, R)
            G = R.T @ self.Sigma @ R @ Lam
            if   mode_Lam == 'oja':     Lam += lr_Lam * G  # Oja's Method (Gradient Ascent)
            elif mode_Lam == 'pm' :     Lam = G.copy()      # Power Method
            Lam -= np.outer(u, np.dot(u, Lam)) / (np.square(u).sum() + 1e-8)     # projection to u = mu_gap' * R
            Lam, _ = np.linalg.qr(Lam)
            V = R @ Lam
            self.evaluate(R=R, V=V)
            if tol is not None:
                _gr = grassmanian_distance(_Lam, Lam)  # compare Lam
                pbar.set_description(f"gap={_gr:.6f}")
                if _gr < tol:
                    print(f"OUTER: Broke in {t} / {n_iter}"); break
        return V
    

    def offline_train_mean(self,
            target_dim,
            n_iter,
            lr,
            seed=None,
            tol=None,
            mode='oja', # ['oja', 'pm']
            lr_scheduler=None,
        ):
        """
        train V in offline manner for mean matching
        """

        ## Check arguments
        assert isinstance(target_dim, int) and target_dim < self.d
        # assert isinstance(rank, int) and rank < self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert isinstance(lr, (int, float)) and lr > 0

        rng = np.random.default_rng(seed)
        # self.r = rank
        self.k = target_dim

        ## Buffers
        self.buffer = Buffer()

        ## Optimization starts from here
        V, _ = np.linalg.qr(rng.standard_normal((self.d, self.k)))    # Optimization variable
        self.evaluate_reg(V=V, rank=0)

        lr0 = lr
        g = self.mu_gap / np.linalg.norm(self.mu_gap)
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _V = V.copy()  # store previous V
            if lr_scheduler is not None:
                lr = lr0 * lr_scheduler(t)
            G = V - np.outer(g, g.dot(self.Sigma @ V))
            if mode == 'oja':   V += lr * G
            else:               V = G
            V, _ = np.linalg.qr(V)
            self.evaluate_reg(V=V)
            if tol is not None:
                _gr = grassmanian_distance(_V, V)  # compare Lam
                pbar.set_description(f"gap={_gr:.6f}")
                if _gr < tol:
                    print(f"OUTER: Broke in {t} / {n_iter}"); break
        return V
    

    def offline_train_reg(self,
            target_dim,
            rank,
            n_iter,
            lr,
            lamda1 = 0.1,
            lamda2 = 0.1,
            seed=None,
            tol=None,
            lr_scheduler=None,
        ):
        """
        train V in offline manner
        """

        ## Check arguments
        assert isinstance(target_dim, int) and target_dim < self.d
        assert isinstance(rank, int) and rank < self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert isinstance(lr, (int, float)) and lr > 0

        rng = np.random.default_rng(seed)
        self.r = rank
        self.k = target_dim

        ## Buffers
        self.buffer = Buffer()

        ## Optimization starts from here
        V, _ = np.linalg.qr(rng.standard_normal((self.d, self.k)))    # Optimization variable
        self.evaluate_reg(V=V, )

        lr0 = lr
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _V = V.copy()  # store previous V
            if lr_scheduler is not None:
                lr = lr0 * lr_scheduler(t)     
            G = self.Sigma @ V

            mean_fairness = self.mu_gap.dot(V)
            # G -= (lamda2/np.linalg.norm(self.Sigma_gap)) * np.outer(self.mu_gap, mean_fairness)
            G -= lamda1 * np.outer(self.mu_gap, mean_fairness)

            cov_fairness = self.Sigma_gap @ V
            # G -= (lamda3/np.linalg.norm(self.Sigma_gap)) * self.Sigma_gap @ cov_fairness
            G -= lamda2 * self.Sigma_gap @ cov_fairness

            V += lr * G
            V, _ = np.linalg.qr(V)
            self.evaluate_reg(V=V)
            if tol is not None:
                _gr = grassmanian_distance(_V, V)  # compare Lam
                pbar.set_description(f"gap={_gr:.6f}, " \
                                     f"meangap={np.linalg.norm(mean_fairness):.6f}, covgap={np.linalg.norm(cov_fairness):.6f}")
                if _gr < tol:
                    print(f"OUTER: Broke in {t} / {n_iter}"); break
        return V
    

    def offline_train_Lam(self,
            target_dim,
            rank,
            n_iter,
            lr,
            seed=None,
            mode='oja',  # ['oja', 'pm']
            tol=None,
            lr_scheduler=None,
        ):
        """
        For a fixed (optimal) R, train Lam in offline manner
        """

        ## Check arguments
        assert isinstance(target_dim, int) and target_dim < self.d
        assert isinstance(rank, int) and target_dim < rank <= self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert isinstance(lr, (int, float)) and lr > 0

        mode = mode.lower()
        assert mode in ['oja', 'pm'], f"Invalid mode: {mode}"
        
        rng = np.random.default_rng(seed)
        self.r = rank  # can be different to self._r
        self.k = target_dim

        ## Buffers
        self.buffer = Buffer()

        ## Optimization starts from here
        Lam, _ = np.linalg.qr(rng.standard_normal((self.r, self.k)))    # Optimization variable
        R = self.eigvec_Sigma_gap_sq[:,:self.r]    # explicitly obtaining R.
        V = R @ Lam
        self.evaluate(R=R, V=V)

        lr0 = lr
        u = np.dot(self.mu_gap, R)
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _Lam = Lam.copy()  # store previous Lam
            if lr_scheduler is not None:
                lr = lr0 * lr_scheduler(t)
            # print(lr)
            G = R.T @ self.Sigma @ R @ Lam
            if   mode == 'oja':     Lam += lr * G  # Oja's Method (Gradient Ascent)
            elif mode == 'pm' :     Lam = G.copy()      # Power Method
            Lam -= np.outer(u, np.dot(u, Lam)) / (np.square(u).sum() + 1e-8)     # projection to u = mu_gap' * R
            Lam, _ = np.linalg.qr(Lam)
            V = R @ Lam
            self.evaluate(R=R, V=V)
            if tol is not None:
                _gr = grassmanian_distance(_Lam, Lam)  # compare Lam
                pbar.set_description(f"gap={_gr:.6f}")
                if _gr < tol:
                    print(f"OUTER: Broke in {t} / {n_iter}"); break
        return V
    

    def offline_train_R(self,
            target_dim,
            rank,
            n_iter,
            mode='square',  # ['square', 'inverse']
            lr=None,
            seed=None,
            tol=None,
            lr_scheduler=None,
        ):
        """
            Training R to have r (==`rank`) following eigenvectors of Sigma_gap^2,
            corresponding to r smallest eigenvalues, in offline manner.
        """
        ## Check arguments
        assert isinstance(rank, int) and rank <= self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert mode in ['square', 'inverse']
        # if lr is not None: assert isinstance(lr, (int, float)) and lr > 0
        # else: 1/self.eigval_Sigma_gap_sq[-1]

        rng = np.random.default_rng(seed)
        self.k = target_dim
        self.r = rank  # can be different to self._r

        ## Buffers
        self.buffer = Buffer()

        self.exp_var_gt, self.V_ground_truth, self.Lam_ground_truth = self.get_ground_truth(self.k, self.r)
        Lam = self.Lam_ground_truth

        ## Optimization starts from here
        R, _ = np.linalg.qr(rng.standard_normal((self.d, self.r)))
        # R = self.eigvec_Sigma_gap_sq[:,:rank]
        if mode == 'inverse': Q_inv = np.linalg.inv(np.eye(self.d) + lr * self.Sigma_gap)
        self.evaluate(V=R@Lam, R=R)

        lr0 = lr
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _R = R.copy() # store previous R
            if lr_scheduler is not None:
                lr = lr0 * lr_scheduler(t)
            G = self.Sigma_gap_sq @ R

            if mode == 'square':    R, _ = np.linalg.qr(R - lr * G)
            elif mode == 'inverse': R, _ = np.linalg.qr(Q_inv @ R)
            
            _, Lam_ = np.linalg.eigh(R.T @ self.Sigma @ R)
            Lam = Lam_[:,-target_dim:]
            # V = R@Lam
            # print(np.linalg.norm(V.T@V - np.eye(target_dim)))

            self.evaluate(V=R@Lam, R=R)
            
            if tol is not None:
                _gr = grassmanian_distance(R, _R)
                pbar.set_description(f"R_gap={_gr:.6f}")
                if _gr < tol:
                    print(f"OUTER: Broke in {t} / {n_iter}"); break
        
        return R
    

    def sample(self, rng=np.random.default_rng(None)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Receive a pair of data `x` and corresponding sensitive attribute `s`

        :: Input
            - rng (np.random.Generator, optional)

        :: Return
            - s (int) : {0, 1}
            - x (np.ndarray) : size=(data_dim, 1)
        """
        s = rng.binomial(1, self.p)
        x = rng.multivariate_normal(eval(f"self.mu{s}"), eval(f"self.Sigma{s}"))
        return s, x


    def train(self,
            target_dim,
            rank,
            n_iter,
            batch_size,
            mean_estimation='local',      # ['local', 'global']
            subspace_optimization='oja',   # ['offline', 'oja', 'history', 'riemannian', 'landing']
            pca_optimization='npm',       # ['offline_oja', 'offline_npm', 'oja', 'npm', 'history', 'riemannian']
            lr_subspace=None,             
            lr_pca=None,                  # must be float when pca_optimization in ['oja']
            n_iter_inner=None,            # must be int when subspace_optimization in ['history'] or pca_optimization in ['history'],
            landing_lambda=None,          # must be float when subspace_optimization == 'landing'
            seed=None,
            tol=1e-6,
            lr_scheduler=None,
            subspace_only=False,
        ):
        """
        Streaminig Fair Block PCA Algorithm
        """
        ## Check arguments
        assert isinstance(target_dim, int) and 0 < target_dim < self.d
        assert isinstance(rank, int) and 0 < rank <= self.d
        assert isinstance(n_iter, int) and n_iter > 0
        assert isinstance(batch_size, int) and batch_size > 0

        assert mean_estimation in ['local', 'global'], mean_estimation
        assert subspace_optimization in ['offline', 'oja', 'history', 'riemannian', 'landing'], subspace_optimization
        assert pca_optimization in ['offline_oja', 'offline_npm', 'oja', 'npm', 'history', 'riemannian']
        assert isinstance(lr_subspace, (int, float)) and lr_subspace > 0
        if pca_optimization in ['oja']:
            assert isinstance(lr_pca, (int, float)) and lr_pca > 0
        if subspace_optimization in ['history'] or pca_optimization in ['history']:
            assert isinstance(n_iter_inner, int) and n_iter_inner > 0
        if subspace_optimization in ['landing']:
            assert isinstance(landing_lambda, float) and landing_lambda > 0

        rng = np.random.default_rng(seed)
        self.r = rank  # can be different to self._r
        self.k = target_dim

        ## Buffers
        self.buffer = Buffer()

        ## Optimization starts from here
        Lam, _ = np.linalg.qr(rng.standard_normal((self.r, self.k)))    # Optimization variable
        R, _ = np.linalg.qr(rng.standard_normal((self.d, self.r)))      # Optimization variable
        if subspace_optimization == 'offline':
            self.exp_var_gt, self.V_ground_truth, self.Lam_ground_truth = self.get_ground_truth(self.k, self.r)
            Lam = self.Lam_ground_truth
        if 'offline' in pca_optimization:
            R = self.eigvec_Sigma_gap_sq[:,:self.r]  # don't need to train R
            u = np.dot(self.mu_gap, R)
        
        V = R @ Lam
        self.evaluate(R=R, V=V)

        n_global = np.zeros(self.num_groups)
        if mean_estimation == 'global': group_mean_global = np.zeros((self.num_groups, self.d))
        if subspace_optimization == 'history': D_R = np.zeros((rank, rank))
        if pca_optimization == 'history': D_Lam = np.zeros((self.k, self.k))
        lr_pca0, lr_subspace0 = lr_pca, lr_subspace
        pbar = trange(1, n_iter+1)
        for t in pbar:
            _V = V.copy()       # store previous V
            _R = R.copy()       # store previous R
            _Lam = Lam.copy()   # store previous Lam
            if lr_scheduler is not None and lr_subspace is not None: lr_subspace = lr_subspace0 * lr_scheduler(t)
            if lr_scheduler is not None and lr_pca is not None: lr_pca = lr_pca0 * lr_scheduler(t)
            
            if subspace_optimization == 'offline':
                R, _ = np.linalg.qr(R - lr_subspace * self.Sigma_gap_sq @ R)
                # _, Lam_ = np.linalg.eigh(R.T @ self.Sigma @ R)
                # Lam = Lam_[:,-target_dim:]
            elif 'offline' not in pca_optimization:
                ## Sampling
                n_local = np.zeros(self.num_groups)
                A, X = [], []
                while n_local.min()<batch_size and not (n_local.sum()>=2*batch_size and n_local.min()>0):
                    s, x = self.sample(rng)
                    A.append(s)
                    X.append(x)
                    n_local[s] += 1
                A = np.array(A)
                X = np.stack(X, 0)
                ## Mean
                group_mean_local = np.zeros((self.num_groups, self.d))
                for s in range(self.num_groups):
                    n_global[s] += n_local[s]
                    group_mean_local[s] = np.mean(X[A==s], 0)
                    if mean_estimation == 'local':
                        X[A==s] -= group_mean_local[s]  # center
                    if mean_estimation == 'global':
                        group_mean_global[s] *= (n_global[s] - n_local[s]) / n_global[s]
                        group_mean_global[s] += (n_local[s]/n_global[s]) * group_mean_local[s]
                        X[A==s] -= group_mean_global[s]  # center
                
                ## Cov
                if subspace_optimization == 'history':
                    for m in range(1,n_iter_inner+1):
                        group_covR = np.zeros((self.num_groups, self.d, self.r))
                        group_covcovR = np.zeros((self.num_groups, self.d, self.r))
                        for s in range(self.num_groups):
                            group_covR[s] = X[A==s].T @ (X[A==s] @ R) / n_local[s]
                        cov_diff_R = group_covR[1] - group_covR[0]
                        for s in range(self.num_groups):
                            group_covcovR[s] = X[A==s].T @ (X[A==s]@ cov_diff_R) / n_local[s] 
                        covcov_diff_R = group_covcovR[1] - group_covcovR[0]
                        S = n_local[s] * (R - lr_subspace * covcov_diff_R)  \
                            + (n_global[s] - n_local[s]) * _R @ D_R @ _R.T @ R
                        S /= n_global[s]
                        R, _ = np.linalg.qr(S)
                    D_R = np.diag(np.linalg.norm(S, axis=0))
                else:
                    group_covR = np.zeros((self.num_groups, self.d, self.r))
                    group_covcovR = np.zeros((self.num_groups, self.d, self.r))
                    for s in range(self.num_groups):
                        group_covR[s] = X[A==s].T @ (X[A==s] @ R) / n_local[s]
                    cov_diff_R = group_covR[1] - group_covR[0]
                    for s in range(self.num_groups):
                        group_covcovR[s] = X[A==s].T @ (X[A==s] @ cov_diff_R) / n_local[s] 
                    covcov_diff_R = group_covcovR[1] - group_covcovR[0]  # Q_hat^2 * R
                    if subspace_optimization == 'oja': 
                        R, _ = np.linalg.qr(R - lr_subspace * covcov_diff_R) 
                    elif subspace_optimization == 'riemannian': 
                        riemannian_grad = .5 * (covcov_diff_R - R @ (covcov_diff_R.T @ R))
                        R, _ = np.linalg.qr(R - lr_subspace * riemannian_grad) 
                    elif subspace_optimization == 'landing': 
                        riemannian_grad = .5 * (covcov_diff_R - R @ (covcov_diff_R.T @ R))
                        field = R @ (R.T @ R) - R
                        R -= lr_subspace * (riemannian_grad + landing_lambda * field)
            if subspace_only:
                self.evaluate(R=R); continue
            
            ## Lam
            if 'offline' in pca_optimization or subspace_optimization == 'offline':
                u = np.dot(self.mu_gap, R)
                if 'history' in pca_optimization: 
                    for m2 in range(1, n_iter_inner+1):
                        S = R.T @ self.Sigma @ R @ Lam
                        S += (t-1) * _Lam @ D_Lam @ _Lam.T @ Lam
                        S /= t
                    D_Lam = np.diag(np.linalg.norm(S, axis=0))
                else:
                    G = R.T @ self.Sigma @ R @ Lam
                    if   'oja' in pca_optimization: Lam += lr_pca * G   # Oja's Method (Gradient Ascent)
                    elif 'npm' in pca_optimization: Lam = G.copy()      # Power Method
                    elif 'riemannian' in pca_optimization: 
                        riemannian_grad = .5 * (G - Lam @ (G.T @ Lam))
                        Lam += lr_pca * riemannian_grad
            else:
                g = X @ R
                if pca_optimization == 'history':
                    for m2 in range(1,n_iter_inner+1):
                        G = g.T @ (g @ Lam) / sum(n_local)
                        S = n_local[s] * G  \
                            + (n_global[s] - n_local[s]) * _Lam @ D_Lam @ _Lam.T @ Lam
                        S /= n_global[s]
                        Lam, _ = np.linalg.qr(S)
                    D_Lam = np.diag(np.linalg.norm(S, axis=0))
                else:
                    G = g.T @ (g @ Lam) / sum(n_local)
                    if pca_optimization == 'oja':
                        Lam += lr_pca * G
                    elif pca_optimization == 'npm':
                        Lam = G.copy()
                    elif subspace_optimization == 'riemannian': 
                        riemannian_grad = .5 * (G - Lam @ (G.T @ Lam))
                        Lam += lr_pca * riemannian_grad
                    # elif subspace_optimization == 'landing': pass # 어떻게 구현하지?
                if mean_estimation == 'local':
                    mean_diff = group_mean_local[1] - group_mean_local[0]
                elif mean_estimation == 'global':
                    mean_diff = group_mean_global[1] - group_mean_global[0]
                u = np.dot(mean_diff, R)
            
            Lam -= np.outer(u, np.dot(u, Lam)) / (np.square(u).sum() + 1e-8)     # projection for equalizing mean
            Lam, _ = np.linalg.qr(Lam)

            V = R @ Lam
            self.evaluate(R=R, V=V)
            if tol is not None:
                _gr = grassmanian_distance(V, _V)
                _nrm = np.sqrt(np.linalg.norm(V.T @ self.Sigma_gap @ V, 2)**2 + np.linalg.norm(self.mu_gap.T @ V)**2)  # fairness constraint
                desc = f"gap={_gr:.6f}, norm_sum={_nrm:.6f}"
                if subspace_optimization == 'history' and 'offline' not in pca_optimization:
                    desc += f", InnerBrk={m}/{n_iter_inner}"
                if pca_optimization == 'history' and subspace_optimization != 'offline':
                    desc += f", InnerBrk2={m2}/{n_iter_inner}"
                pbar.set_description(desc)
                if _gr < tol:
                    print(f"OUTER: Broke in {t} / {n_iter}"); break
        return V


    def transform(self, x:np.ndarray, V=None) -> np.ndarray:
        if V is None:
            assert hasattr(self, 'V'), 'Training is NOT done'
            V = self.V
        return np.dot(V, np.dot(V, x))
    

def get_args():
    parser = ArgumentParser()
    
    ## Data Generation
    parser.add_argument('-d', '--data_dim', type=int, default=100)
    parser.add_argument('-p', '--probability', type=float, default=0.5)
    parser.add_argument('-R', '--rank_true', type=int, default=10)
    parser.add_argument('--seed_data', type=int, default=None)
    parser.add_argument('--mu_scale', type=float, default=1.)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--max_cov_eigs', nargs=2, type=float, default=[1., 2.])

    ## PCA
    parser.add_argument('-k', '--target_dim', type=int, default=3)
    parser.add_argument('-r', '--rank', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--mean_estimation', type=str, default='local')     # ['local', 'global']
    parser.add_argument('--subspace_optimization', type=str, default='oja')  # ['oja', 'history']
    parser.add_argument('--pca_optimization', type=str, default='npm')      # ['oja', 'npm', 'history']

    parser.add_argument('--lr_subspace', type=float, default=None)           # must be float when pca_optimization in ['oja']
    parser.add_argument('--lr_pca', type=float, default=None)                # must be int when subspace_optimization in ['history'] or pca_optimization in ['history']
    parser.add_argument('--landing_lambda', type=float, default=None)        # must be float when subspace_optimization == 'landing' 
    parser.add_argument('--n_iter_inner', type=int, default=None)
    parser.add_argument('--seed_train', type=int, default=None)
    parser.add_argument('--tolerance', type=float, default=None)
    parser.add_argument('--subspace_only', action='store_true')
    args = parser.parse_args()
    return args


def run(args):
    Algo = StreamingFairBlockPCA(
        data_dim=args.data_dim,
        num_attributes=1,
        num_groups=2,
        probability=args.probability,
        rank=args.rank_true,
        seed=args.seed_data,
        mu_scale=args.mu_scale,
        eps=args.eps,
        max_cov_eig0=args.max_cov_eigs[0],
        max_cov_eig1=args.max_cov_eigs[1],
    )

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

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
        mean_estimation=args.mean_estimation,
        subspace_optimization=args.subspace_optimization,
        pca_optimization=args.pca_optimization,
        lr_subspace=args.lr_subspace, 
        lr_pca=args.lr_pca,
        n_iter_inner=args.n_iter_inner,
        landing_lambda=args.landing_lambda,
        seed=args.seed_train,
        tol=args.tolerance,
        # lr_scheduler=None,
        subspace_only=args.subspace_only,
    )

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax[0][0].plot(Algo.buffer.explained_variance_ratio)
    ax[0][0].hlines(Algo.exp_var_gt/Algo.total_var, 0, args.n_iter+1, colors='r')
    ax[0][0].set_title('Explained Variance Ratio')

    ax[0][1].plot(Algo.buffer.objective_unfairness)
    # ax[0][1].hlines(Algo.exp_var_gt/Algo.total_var, 0, args.n_iter+1, colors='r')
    ax[0][1].set_title('Objective unfairness')

    ax[0][2].plot(Algo.buffer.distance_from_gt)
    ax[0][2].set_title('Grassmanian distance to $V^\\ast$')

    ax[0][3].plot(Algo.buffer.R_dist)
    ax[0][3].set_title('$|| RR^T - R_{opt}R_{opt}^T ||$')

    ax[1][0].plot(Algo.buffer.projected_mean_diff)
    ax[1][0].set_title('Projected_mean_gap (mean fairness)')

    ax[1][1].plot(Algo.buffer.projected_cov_diff)
    ax[1][1].set_title('Projected_cov_gap (Covar. fairness)')

    ax[1][2].plot(Algo.buffer.projected_R_norm)
    ax[1][2].set_title("$tr(R^T \Sigma_{gap} R)$")
    # fig.show()
    fig.savefig('records/run_info.pdf', bbox_inches='tight')


if __name__ == "__main__":
    run(get_args())
    