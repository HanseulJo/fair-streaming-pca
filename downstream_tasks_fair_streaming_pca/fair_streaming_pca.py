import jax.numpy as jnp
from jax import random

class FairPCA:
    def __init__(
            self, 
        ) -> None:
        self.f = jnp.nan   # Unfair subspace parameter (Mean)
        self.W = jnp.nan   # Unfair subspace parameter (Covariance)
        self.N = jnp.nan   # Unfair subspace parameter (Covariance)
        self.V = jnp.nan   # Fair PCA parameter

    @property
    def unfair_subspace_mean(self):
        return self.f

    @property
    def unfair_subspace_cov(self):
        return self.W
    
    @property
    def unfair_subspace_all(self):
        return self.N
    
    @property
    def fair_principal_component(self):
        return self.V
    

    ###############################################
    ##  Fit Offline (direct Eigendecomposition)  ##
    ###############################################
    def fit_offline(self,
            X, A,
            target_unfair_dim,
            target_pca_dim,
            constraint='all',
            fairness_tradeoff=1.,
            # verbose=False,
            ) -> None:
        
        self.k = target_pca_dim
        self.m = target_unfair_dim
        self.d = X.shape[-1]
        self.constraint = constraint
        self.fairness_tradeoff = fairness_tradeoff
        # self.verbose = verbose
        X = jnp.asarray(X)
        A = jnp.asarray(A)
        
        ## SUBSPACE ESTIMATION
        if constraint in ['mean', 'covariance', 'all']:
            self._unfair_subspace_estimation_offline(X, A)
        ## PCA OPTIMIZATION
        self._principal_component_analysis_offline(X)

        ## evaluate explained variance
        self.eigval_Sigma, self.eigvec_Sigma = jnp.linalg.eigh(self.Sigma)
        optimal_explained_variance = self.eigval_Sigma.sum()
        self.explained_variance_ratio = jnp.trace(self.V.T @ self.Sigma @ self.V) / optimal_explained_variance

    def _unfair_subspace_estimation_offline(self, X, A):
        if self.constraint in ['mean', 'all']:
            self.mean_group = [jnp.mean(X[A==a],0) for a in range(2)]
            self.f = self.mean_group[1] - self.mean_group[0]
            self.f /= (jnp.linalg.norm(self.f) + 1e-8)
            if self.constraint == 'mean':
                self.N = jnp.expand_dims(self.f, -1)
        
        if self.constraint in ['covariance', 'all']:
            self.cov_group = [jnp.cov(X[A==a].T) for a in range(2)]
            # self.cov_group = [X[A==a].T @ X[A==a] / (X[A==a].shape[0]) for a in range(2)]
            self.Q = self.cov_group[1]-self.cov_group[0]
            eigval_Q, eigvec_Q = jnp.linalg.eigh(self.Q)
            _indices_W = jnp.argsort(jnp.abs(eigval_Q))[-self.m:][::-1]
            self.W = eigvec_Q[:,_indices_W]
            if self.constraint == 'covariance':
                self.N = self.W.copy()

        if self.constraint == 'all':
            ## Normal subspace; for both mean & covariance gap
            if jnp.isclose(jnp.linalg.norm(self.f), 0):
                self.N = self.W.copy()
            elif jnp.isclose(jnp.linalg.norm(self.W), 0):
                self.N = jnp.expand_dims(self.f, -1)
            else:
                g = self.f - self.W @ self.W.T @ self.f
                g /= (jnp.linalg.norm(g) + 1e-8)
                self.N = jnp.concatenate([self.W, jnp.expand_dims(g, -1)], -1)
                # self.N, _ = jnp.linalg.qr(self.N)

        if jnp.isnan(self.N).any():
            raise ValueError('nan N')

    def _principal_component_analysis_offline(self, X):
        self.Sigma = jnp.cov(X.T)
        if self.constraint in ['mean', 'covariance', 'all']:
            Pi_N = jnp.eye(self.d) - self.fairness_tradeoff * self.N @ self.N.T
            Sigma_ = Pi_N @ self.Sigma @ Pi_N
        else:
            Sigma_ = self.Sigma
        eigval, eigvec = jnp.linalg.eigh(Sigma_)
        _indices_V = jnp.argsort(jnp.abs(eigval))[-self.k:][::-1]
        self.V = eigvec[:,_indices_V]

    
    ####################################
    ##  Fit Streaming (power method)  ##
    ####################################
    def fit_streaming(self, 
            X, A,
            target_unfair_dim,
            target_pca_dim,
            n_iter_unfair,
            n_iter_pca,
            constraint='all',
            fairness_tradeoff=1.,
            seed=None,
            # verbose=False,
            ) -> None:
        
        self.k = target_pca_dim
        self.m = target_unfair_dim
        self.d = X.shape[-1]
        self.n_iter_unfair = n_iter_unfair
        self.n_iter_pca = n_iter_pca
        self.constraint = constraint
        self.fairness_tradeoff = fairness_tradeoff
        # self.verbose = verbose
        X = jnp.asarray(X)
        A = jnp.asarray(A)
        if seed is None:
            from sys import maxsize
            import numpy as np
            seed = np.random.randint(0,maxsize)
        key = random.PRNGKey(seed)
        
        # Initialization of parameters
        key, rng = random.split(key)
        self.V, _ = jnp.linalg.qr(random.normal(rng, (self.d, self.k)))
        if constraint in ['covariance', 'all']:
            key, rng = random.split(key)
            self.W, _ = jnp.linalg.qr(random.normal(rng, (self.d, self.m)))
        
        ## SUBSPACE ESTIMATION
        if constraint in ['mean', 'covariance', 'all']:
            self._unfair_subspace_estimation_pm(X, A)
        ## PCA OPTIMIZATION
        self._principal_component_analysis_pm(X)

        ## evaluate explained variance
        self.eigval_Sigma, self.eigvec_Sigma = jnp.linalg.eigh(self.Sigma)
        optimal_explained_variance = self.eigval_Sigma.sum()
        self.explained_variance_ratio = jnp.trace(self.V.T @ self.Sigma @ self.V) / optimal_explained_variance
    
    def _unfair_subspace_estimation_pm(self, X, A) -> None:
        if self.constraint in ['mean', 'all']:
            self.mean_group = [jnp.mean(X[A==a],0) for a in range(2)]
            self.f = self.mean_group[1] - self.mean_group[0]
            self.f /= (jnp.linalg.norm(self.f) + 1e-8)
            if self.constraint == 'mean':
                self.N = jnp.expand_dims(self.f, -1)
        
        if self.constraint in ['covariance', 'all']:
            # self.cov_group = [jnp.cov(X[A==a].T) for a in range(2)]
            self.cov_group = [X[A==a].T @ X[A==a] / (X[A==a].shape[0]) for a in range(2)]
            self.Q = self.cov_group[1]-self.cov_group[0]
            for _ in range(self.n_iter_unfair):
                self.W, _ = jnp.linalg.qr(self.Q @ self.W)
            if self.constraint == 'covariance':
                self.N = self.W.copy()

        if self.constraint == 'all':
            ## Normal subspace; for both mean & covariance gap
            if jnp.isclose(jnp.linalg.norm(self.f), 0):
                self.N = self.W.copy()
            elif jnp.isclose(jnp.linalg.norm(self.W), 0):
                self.N = jnp.expand_dims(self.f, -1)
            else:
                g = self.f - self.W @ self.W.T @ self.f
                g /= (jnp.linalg.norm(g) + 1e-8)
                self.N = jnp.concatenate([self.W, jnp.expand_dims(g, -1)], -1)
                # self.N, _ = jnp.linalg.qr(self.N)
        
        if jnp.isnan(self.N).any():
            raise ValueError('nan N')

    def _principal_component_analysis_pm(self, X) -> None:
        self.Sigma = jnp.cov(X.T)
        for _ in range(self.n_iter_pca):
            if self.constraint in ['mean', 'covariance', 'all']:
                self.V -= self.fairness_tradeoff * self.N @ self.N.T @ self.V
            self.V = self.Sigma @ self.V
            if self.constraint in ['mean', 'covariance', 'all']:
                self.V -= self.fairness_tradeoff * self.N @ self.N.T @ self.V
            self.V, _ = jnp.linalg.qr(self.V)

    ################
    ## Projection ##
    ################
    def transform(self, x):
        if x.shape == (self.d,):
            result = self.V @ self.V.T @ x
        else:
            result = self.V @ self.V.T @ x.T
            result = result.T
        return result
    
    def transform_low_dim(self, x):
        if x.shape == (self.d,):
            result = self.V.T @ x
        else:
            result = self.V.T @ x.T
            result = result.T
        return result
