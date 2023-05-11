
import jax.numpy as jnp
from jax import random
import numpy as np
import torch
from tqdm.auto import tqdm, trange

from utils import mmd


ATTR = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
ATTR = {attr: attr_idx for attr_idx, attr in enumerate(ATTR.split(' '))}


class Buffer:
    """
    Arbitrary Memory buffer to record performance of PCA algorithm.
    """
    def __init__(self):
        self.explained_variance_ratio = []
        self.explained_variance_ratio_0 = []
        self.explained_variance_ratio_1 = []
        self.maximum_mean_discrepancy = []
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k).append(v)

class FairStreamingPCA:
    """
    """
    def __init__(
            self, 
            target_attr: str,
        ) -> None:
        self.a = ATTR[target_attr]
        self.f = None   # Unfair subspace parameter (Mean)
        self.W = None   # Unfair subspace parameter (Covariance)
        self.N = None   # Unfair subspace parameter (Covariance)
        self.V = None   # Fair PCA parameter

    @property
    def target_attribute_index(self):
        return self.a

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

    def fit(self, 
            loader,
            target_unfair_dim,
            target_pca_dim,
            n_iter_unfair,
            n_iter_pca,
            block_size_unfair,
            block_size_pca,
            constraint='all',
            unfairness_optimization='npm',
            pca_optimization='npm',
            unfairness_frequent_direction=False,
            pca_frequent_direction=False,
            fairness_tradeoff=1.,
            seed=None,
            logger=None,
            verbose=False) -> None:
        
        self.k = target_pca_dim
        self.m = target_unfair_dim
        self.batch_size, self.num_channel, height, width = next(iter(loader))[0].size()
        self.d = height * width
        if seed is None:
            from sys import maxsize
            seed = np.random.randint(0,maxsize)
        self.key = random.PRNGKey(seed)

        self.n_iter_unfair = n_iter_unfair
        self.n_iter_pca = n_iter_pca
        self.block_size_unfair = block_size_unfair
        self.block_size_pca = block_size_pca
        self.constraint = constraint
        self.unfairness_optimization = unfairness_optimization
        self.pca_optimization = pca_optimization
        self.unfairness_frequent_direction=unfairness_frequent_direction
        self.pca_frequent_direction=pca_frequent_direction
        self.fairness_tradeoff = fairness_tradeoff
        self.verbose = verbose

        # Recording performances
        # self.buffer = Buffer()

        # Initialization of parameters
        self.key, rng = random.split(self.key)
        self.V, _ = jnp.linalg.qr(random.normal(rng, (self.num_channel, self.d, self.k)))
        if constraint in ['covariance', 'all']:
            self.key, rng = random.split(self.key)
            self.W, _ = jnp.linalg.qr(random.normal(rng, (self.num_channel, self.d, self.m)))
        ## SUBSPACE ESTIMATION
        if constraint in ['mean', 'covariance', 'all']:
            self._unfair_subspace_estimation_with_npm(loader)
        ## PCA OPTIMIZATION
        self._principal_component_analysis_with_npm(loader)

    def _unfair_subspace_estimation_with_npm(self, loader) -> None:
        self.b_global_group = [0 for _ in range(2)]
        self.mean_global_group = [jnp.zeros((self.num_channel, self.d)) for _ in range(2)]
        self.B_W = None if not self.unfairness_frequent_direction else jnp.zeros((self.num_channel, self.d, 2*self.m))

        for t in trange(1, self.n_iter_unfair+1):
            self._unfair_covariance_estimation_with_npm(loader, t)

        if self.constraint in ['mean', 'all']:
            b_global = sum(self.b_global_group)
            self.f = (self.b_global_group[0] * self.mean_global_group[1] -  self.b_global_group[1] * self.mean_global_group[0]) / b_global
            self.f /= (jnp.linalg.norm(self.f) + 1e-8)
            if self.constraint == 'mean':
                self.N = self.f.reshape(-1, 1)

        elif self.constraint == 'covariance':
            self.N = self.W.copy()
                        
        if self.constraint == 'all':
            ## Normal subspace; for both mean & covariance gap
            if jnp.isclose(jnp.linalg.norm(self.f), 0):
                self.N = self.W.copy()
            else:
                self.N = jnp.concatenate([jnp.expand_dims(self.f, -1), self.W], -1)
                self.N, _ = jnp.linalg.qr(self.N)


    def _unfair_covariance_estimation_with_npm(self, loader, t):
        b_local_group = [0 for _ in range(2)]
        if self.constraint in ['mean', 'all']:
            mean_local_group = [jnp.zeros((self.num_channel, self.d)) for _ in range(2)]
        if self.constraint in ['covariance', 'all']:
            cov_W_group = [jnp.zeros((self.num_channel, self.d, self.m)) for _ in range(2)]
        
        _n_iter = self.block_size_unfair // self.batch_size
        b_local_group = [0 for _ in range(2)]
        pbar = tqdm(enumerate(loader), total=_n_iter, disable=not self.verbose, desc=f'UnfairCovEstim({t}/{self.n_iter_unfair}):')
        # pbar = enumerate(loader)
        for batch_index, (img, label) in pbar:
            if batch_index >= _n_iter: break
            x = jnp.asarray(img).reshape(self.batch_size, self.num_channel, self.d)
            ss = jnp.asarray(label[...,self.a])
            xs = [[], []]
            for i, s in enumerate(ss):
                xs[s].append(x[i])
                b_local_group[s] += 1
            for s in range(2):
                if len(xs[s]) == 0: continue
                xs[s] = jnp.stack(xs[s])
                if self.constraint in ['mean', 'all']:
                    mean_local_group[s] += xs[s].sum(0) / self.block_size_unfair
                if self.constraint in ['covariance', 'all']:
                    cov_W_group[s] += jnp.einsum("bcd,bcm->cdm", xs[s], jnp.einsum("bcd,cdm->bcm", xs[s], self.W)) / self.block_size_unfair
            
        for s in range(2):
            if self.constraint in ['mean', 'all']:
                self.mean_global_group[s] *= self.b_global_group[s]
                self.mean_global_group[s] += b_local_group[s] * mean_local_group[s]
                self.mean_global_group[s] /= self.b_global_group[s] + b_local_group[s]
            self.b_global_group[s] += b_local_group[s]

        if self.constraint in ['covariance', 'all']:
            ## Covariance gap
            covdiff_W = (b_local_group[0] * cov_W_group[1] - b_local_group[1] * cov_W_group[0]) / self.batch_size
            if self.unfairness_optimization == 'npm': 
                self.W, _ = jnp.linalg.qr(covdiff_W)
            else: raise NotImplementedError
            if self.unfairness_frequent_direction:
                self.B_W = self.B_W.at[...,-self.m:].set(self.W)
                U, S, _ = jnp.linalg.svd(self.B_W, full_matrices=False)  # singular value in decreasing order
                min_singular = jnp.expand_dims(jnp.square(S[...,self.m]),-1)
                S = jnp.sqrt(jnp.clip(jnp.square(S)-min_singular, 0, None))
                S = jnp.stack([jnp.diag(S_) for S_ in S])
                self.B_W = jnp.einsum("cdm,cmn->cdn", U, S)
                self.W, _ = jnp.linalg.qr(self.B_W[...,:self.m])
            

    def _principal_component_analysis_with_npm(self, loader) -> None:
        b_global = 0
        mean_global = jnp.zeros((self.num_channel, self.d))
        self.B_V = None
        if self.pca_frequent_direction: self.B_V = jnp.zeros((self.num_channel, self.d, 2*self.k))
        
        for t in trange(1, self.n_iter_pca+1):
            # _V = self.V.copy()       # store previous V

            ## Before Sampling
            mean_local = jnp.zeros((self.num_channel, self.d))
            cov_V =  jnp.zeros((self.num_channel, self.d,self.k))
            if self.constraint in ['mean', 'covariance', 'all']:
                ## projection
                self.V -= self.fairness_tradeoff * jnp.einsum("cdm,cmk->cdk", self.N, jnp.einsum("cdm,cdk->cmk",self.N, self.V))

            ## Sampling
            _n_iter = self.block_size_pca // self.batch_size
            b_local = 0
            # pbar = tqdm(enumerate(loader), total=_n_iter)
            pbar = tqdm(enumerate(loader), total=_n_iter, disable=not self.verbose, desc=f'PCA({t}/{self.n_iter_pca}):')
            for batch_index, (img, _) in pbar:
                if batch_index >= _n_iter: break
                x = jnp.asarray(img).reshape(self.batch_size, self.num_channel, self.d)
                mean_local += x.sum(0) / self.block_size_pca
                cov_V += jnp.einsum("bcd,bcm->cdm", x-x.mean(0), jnp.einsum("bcd,cdm->bcm", x-x.mean(0), self.V)) / self.block_size_pca
                b_local += self.batch_size

            ## After Sampling
            mean_global *= b_global
            mean_global += self.block_size_pca * mean_local
            mean_global /= b_global + b_local
            b_global += b_local
            if self.constraint in ['mean', 'covariance', 'all']:
                ## projection
                cov_V -= self.fairness_tradeoff * jnp.einsum("cdm,cmk->cdk", self.N, jnp.einsum("cdm,cdk->cmk",self.N, self.V))
            
            if self.pca_optimization == 'npm':
                self.V, _ = jnp.linalg.qr(cov_V)
            else: raise NotImplementedError
            if self.pca_frequent_direction:
                self.B_V = self.B_V.at[...,-self.k:].set(self.V)
                U, S, _ = jnp.linalg.svd(self.B_V, full_matrices=False)  # singular value in decreasing order
                min_singular = np.expand_dims(jnp.square(S[...,self.k]),-1)
                S = jnp.sqrt(jnp.clip(jnp.square(S)-min_singular, 0, None))
                S = jnp.stack([jnp.diag(S_) for S_ in S])
                self.B_V = jnp.einsum("cdm,cmn->cdn", U, S)
                self.V, _ = jnp.linalg.qr(self.B_V[...,:self.k])

    def transform(self, x=None, loader=None, lambda_transform=lambda t: t):
        assert self.V is not None
        assert (x is None and loader is not None) or (x is not None and loader is None)
        if x is not None:
            arr = jnp.asarray(x)
            if arr.ndim == 3:
                arr = jnp.expand_dims(arr,0)
            bs, num_ch, h, w = arr.shape
            arr = arr.reshape(bs, num_ch, h*w)
            proj_arr = jnp.einsum("cdk,bck->bcd", self.V, jnp.einsum("cdk,bcd->bck", self.V, arr))
            result = proj_arr.reshape(bs, num_ch, h, w)
            return lambda_transform(result)
        elif loader is not None:
            batch_size, num_channel, height, width = next(iter(loader))[0].size()
            b_local_group = [0 for _ in range(2)]
            pbar = tqdm(loader, desc=f'Evaluation:')
            xs = [[], []]
            xs_transfomed = [[],[]]
            self.explained_variance_ratio_group = [0, 0]

            for img, label in pbar:
                x = jnp.asarray(img)  # (batch_size, num_channel, height, width)
                ss = jnp.asarray(label[...,self.a])
                for i, s in enumerate(ss):
                    xs[s].append(x[i])
                    b_local_group[s] += 1
            for s in range(2):
                xs[s] = jnp.stack(xs[s])
                xs_transfomed[s] = self.transform(xs[s], lambda_transform= lambda x: x.reshape(batch_size, num_channel, self.d))
                self.explained_variance_ratio_group[s] = self._get_explained_variance_ratio(xs[s], xs_transfomed[s])
            X = jnp.concatenate(xs)
            X_transfomed = jnp.concatenate(xs_transfomed)
            self.explained_variance_ratio = self._get_explained_variance_ratio(X, X_transfomed)
            self.maximum_mean_discrepancy = mmd(xs[0].reshape(batch_size, -1), xs[1].reshape(batch_size, -1))

    def _get_explained_variance_ratio(self, x, x_transformed=None):
        var = lambda X : jnp.trace(jnp.einsum("bcd,bcd->bcc", X, X))
        if x_transformed is None:
            x_transformed = self.transform(x)
        return var(x_transformed) / var(x)
    
    def fit_transform(self, loader_train, x, seed=None):
        self.fit(loader_train,seed=seed)
        result = self.transform(x)
        return result
