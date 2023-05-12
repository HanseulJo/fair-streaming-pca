
import numpy as np
import jax.numpy as jnp
import torch
from tqdm.auto import tqdm, trange
from scipy.linalg import null_space

# from kleindessner.src.fair_pca.fair_PCA import (
#     solve_standard_eigenproblem_for_smallest_magnitude_eigenvalues,
#     solve_standard_eigenproblem_for_largest_eigenvalues
# )


ATTR = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
ATTR = {attr: attr_idx for attr_idx, attr in enumerate(ATTR.split(' '))}

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'

class FairStreamingPCA:
    """
    """
    def __init__(
            self, 
            target_attr: str,
            device: str = 'cpu'
        ) -> None:
        self.a = ATTR[target_attr]
        self.f = torch.empty(1)   # Unfair subspace parameter (Mean)
        self.W = torch.empty(1)   # Unfair subspace parameter (Covariance)
        self.N = torch.empty(1)   # Unfair subspace parameter (Covariance)
        self.V = torch.empty(1)   # Fair PCA parameter
        self.device = device

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
            # logger=None,
            verbose=False,
            loader_val=None) -> None:
        
        self.k = target_pca_dim
        self.m = target_unfair_dim
        self.batch_size, self.num_channel, height, width = next(iter(loader))[0].size()
        self.d = height * width
        if seed is not None:
            torch.manual_seed(seed)

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
        self.buffer_V = []

        # Initialization of parameters
        self.V, _ = torch.linalg.qr(torch.normal(0, 1, size=(self.num_channel, self.d, self.k)))
        self.V = self.V.to(self.device)
        if constraint in ['covariance', 'all']:
            self.W, _ = torch.linalg.qr(torch.normal(0, 1, size=(self.num_channel, self.d, self.m)))
            self.W = self.W.to(self.device)
        
        ## SUBSPACE ESTIMATION
        if constraint in ['mean', 'covariance', 'all']:
            self._unfair_subspace_estimation(loader, loader_val=loader_val)
        ## PCA OPTIMIZATION
        self._principal_component_analysis(loader)

    def _unfair_subspace_estimation(self, loader, loader_val=None) -> None:
        self.b_global_group = [0 for _ in range(2)]
        self.mean_global_group = [torch.zeros(self.num_channel, self.d).to(self.device) for _ in range(2)]
        self.B_W = None if not self.unfairness_frequent_direction else torch.zeros(self.num_channel, self.d, 2*self.m).to(self.device)

        pbar = trange(1, self.n_iter_unfair+1, desc="UnfairEstim:")
        for t in pbar:
            self._unfair_subspace_estimation_with_npm(loader, t)
            if torch.isnan(self.W).any():
                raise ValueError('nan W')

        if self.constraint in ['mean', 'all']:
            b_global = sum(self.b_global_group)
            self.f = (self.b_global_group[0] * self.mean_global_group[1] -  self.b_global_group[1] * self.mean_global_group[0]) / b_global
            if torch.isnan(self.f).any():
                raise ValueError('nan f before normalization')
            self.f /= (torch.linalg.norm(self.f.cpu()) + 1e-8)
            if self.constraint == 'mean':
                self.N = torch.unsqueeze(self.f, -1)

        elif self.constraint == 'covariance':
            self.N = self.W.clone()
                        
        if self.constraint == 'all':
            ## Normal subspace; for both mean & covariance gap
            if torch.isclose(torch.norm(self.f), torch.zeros(1).to(self.device)):
                self.N = self.W.clone()
            elif torch.isclose(torch.norm(self.W), torch.zeros(1).to(self.device)):
                self.N = torch.unsqueeze(self.f, -1)
            else:
                self.N = torch.cat([torch.unsqueeze(self.f, -1), self.W], -1)
                self.N, _ = torch.linalg.qr(self.N.cpu())
                self.N = self.N.to(self.device)

        
        if torch.isnan(self.N).any():
            raise ValueError('nan N')


    def _unfair_subspace_estimation_with_npm(self, loader, t):
        b_local_group = [0 for _ in range(2)]
        if self.constraint in ['mean', 'all']:
            mean_local_group = [torch.zeros(self.num_channel, self.d).to(self.device) for _ in range(2)]
        if self.constraint in ['covariance', 'all']:
            cov_W_group = [torch.zeros(self.num_channel, self.d, self.m).to(self.device) for _ in range(2)]
        
        _n_iter = self.block_size_unfair // self.batch_size
        b_local_group = [0 for _ in range(2)]
        pbar = tqdm(enumerate(loader), total=_n_iter, disable=not self.verbose, desc=f'UnfairEstim({t}/{self.n_iter_unfair}):')
        # pbar = enumerate(loader)
        for batch_index, (img, label) in pbar:
            if batch_index >= _n_iter: break
            img, label = img.to(self.device), label.to(self.device)
            x = img.view(self.batch_size, self.num_channel, self.d)
            ss = label[...,self.a]
            for s in range(2):
                if self.constraint in ['mean', 'all']:
                    mean_local_group[s] += x[ss==s].sum(0) / self.block_size_unfair
                if self.constraint in ['covariance', 'all']:
                    cov_W_group[s] += torch.einsum("bcd,bcm->cdm", x[ss==s], torch.einsum("bcd,cdm->bcm", x[ss==s], self.W)) / self.block_size_unfair
                b_local_group[s] += (ss==s).sum()
        
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
                self.W, _ = torch.linalg.qr(covdiff_W.cpu())
                self.W = self.W.to(self.device)
            else: raise NotImplementedError
            if self.unfairness_frequent_direction:
                self.B_W[...,-self.m:] = self.W
                U, S, _ = torch.linalg.svd(self.B_W.cpu(), full_matrices=False)  # singular value in decreasing order
                min_singular = torch.unsqueeze(torch.square(S[...,self.m]),-1)
                S = torch.sqrt(torch.clip(torch.square(S)-min_singular, 0, None))
                S = torch.stack([torch.diag(S_) for S_ in S])
                self.B_W = torch.einsum("cdm,cmn->cdn", U, S).to(self.device)
                self.W, _ = torch.linalg.qr(self.B_W[...,:self.m].cpu())
                self.W = self.W.to(self.device)

    def _principal_component_analysis(self, loader) -> None:
        b_global = 0
        mean_global = torch.zeros(self.num_channel, self.d).to(self.device)
        self.B_V = None
        if self.pca_frequent_direction: self.B_V = torch.zeros(self.num_channel, self.d, 2*self.k).to(self.device)
        
        self.buffer_V.append(self.V.clone())

        for t in trange(1, self.n_iter_pca+1,  desc=f'PCA:'):
            ## Before Sampling
            mean_local = torch.zeros(self.num_channel, self.d).to(self.device)
            cov_V =  torch.zeros(self.num_channel, self.d,self.k).to(self.device)
            if self.constraint in ['mean', 'covariance', 'all']:
                ## projection
                self.V -= self.fairness_tradeoff * torch.einsum("cdm,cmk->cdk", self.N, torch.einsum("cdm,cdk->cmk",self.N, self.V))
                if torch.isnan(self.V).any():
                    raise ValueError('nan V')
            ## Sampling
            _n_iter = self.block_size_pca // self.batch_size
            b_local = 0
            # pbar = tqdm(enumerate(loader), total=_n_iter)
            pbar = tqdm(enumerate(loader), total=_n_iter, disable=not self.verbose, desc=f'PCA({t}/{self.n_iter_pca}):')
            for batch_index, (img, _) in pbar:
                img = img.to(self.device)
                if batch_index >= _n_iter: break
                x = img.view(self.batch_size, self.num_channel, self.d)
                mean_local += x.sum(0) / self.block_size_pca
                cov_V += torch.einsum("bcd,bcm->cdm", x, torch.einsum("bcd,cdm->bcm", x, self.V)) / self.block_size_pca
                b_local += self.batch_size

            ## After Sampling
            mean_global *= b_global
            mean_global += self.block_size_pca * mean_local
            mean_global /= b_global + b_local
            b_global += b_local
            if self.constraint in ['mean', 'covariance', 'all']:
                ## projection
                cov_V -= self.fairness_tradeoff * torch.einsum("cdm,cmk->cdk", self.N, torch.einsum("cdm,cdk->cmk",self.N, cov_V))
            if self.pca_optimization == 'npm':
                self.V, _ = torch.linalg.qr(cov_V.cpu())
                self.V = self.V.to(self.device)
            else: raise NotImplementedError
            if self.pca_frequent_direction:
                self.B_V[...,-self.k:] = self.V
                U, S, _ = torch.linalg.svd(self.B_V.cpu(), full_matrices=False)  # singular value in decreasing order
                min_singular = torch.unsqueeze(torch.square(S[...,self.k]),-1)
                S = torch.sqrt(torch.clip(torch.square(S)-min_singular, 0, None))
                S = torch.stack([torch.diag(S_) for S_ in S])
                self.B_V = torch.einsum("cdm,cmn->cdn", U, S).to(self.device())
                self.V, _ = torch.linalg.qr(self.B_V[...,:self.k].cpu())
                self.V = self.V.to(self.device)
            
            if t % 1 == 0:
                self.buffer_V.append(self.V.clone())
            

    def transform(self, x=None, loader=None, lambda_transform=lambda t: t):
        assert self.V is not None
        assert (x is None and loader is not None) or (x is not None and loader is None)
        if x is not None:
            x = x.to(self.device)
            if x.ndim == 3:
                x = torch.unsqueeze(x,0)
            bs, num_ch, h, w = x.size()
            x = x.view(bs, num_ch, h*w)
            proj_arr = torch.einsum("cdk,bck->bcd", self.V, torch.einsum("cdk,bcd->bck", self.V, x))
            result = proj_arr.view(bs, num_ch, h, w)
            return lambda_transform(result)
        elif loader is not None:
            # batch_size, num_channel, height, width = next(iter(loader))[0].size()
            b = 0
            b_group = [0 for _ in range(2)]
            self.explained_variance_ratio = 0
            self.explained_variance_ratio_group = [0,0]
            pbar = tqdm(loader, desc=f'Evaluation:')
            for img, label in pbar:
                x = img.to(self.device)  # (batch_size, num_channel, height, width)
                ss = label[...,self.a]
                for s in range(2):
                    self.explained_variance_ratio_group[s] += self._get_explained_variance_ratio(x[ss==s]) * (ss==s).sum().item()
                    b_group[s] += (ss==s).sum().item()
                self.explained_variance_ratio += self._get_explained_variance_ratio(x) * x.size(0)
                b += x.size(0)
            for s in range(2):
                self.explained_variance_ratio_group[s] /= b_group[s]
            self.explained_variance_ratio /= b

    def transform_low_dimension(self, x=None, loader=None, lambda_transform=lambda t: t):
        assert self.V is not None
        assert (x is None and loader is not None) or (x is not None and loader is None)
        if x is not None:
            x = x.to(self.device)
            if x.ndim == 3:
                x = torch.unsqueeze(x,0)
            bs, num_ch, h, w = x.size()
            x = x.view(bs, num_ch, h*w)
            proj_arr = torch.einsum("cdk,bcd->bck", self.V, x)
            result = proj_arr.view(bs, num_ch, h, w)
            return lambda_transform(result)

    
    def _get_explained_variance_ratio(self, x, x_transformed=None):
        if x_transformed is None:
            x_transformed = self.transform(x)
        x = x.view(-1, self.num_channel, self.d)
        x_transformed = x_transformed.view(-1, self.num_channel, self.d)
        var = torch.norm(x, dim=(0,2))
        var_transformed = torch.norm(x_transformed, dim=(0,2))
        return (var_transformed / var).mean().item()

    def fit_offline(self, 
            loader,
            target_unfair_dim,
            target_pca_dim,
            constraint='all',
        ):

        batch_size, self.num_channel, height, width = next(iter(loader))[0].size()
        self.d = height*width
        b_local_group = [0 for _ in range(2)]
        Xs = [[], []]
        Ss = []
        self.explained_variance_ratio_group = [0, 0]
        pbar = tqdm(loader, desc=f'Data Collection:')
        for img, label in pbar:
            x = img.view(batch_size, self.num_channel, -1)  # (batch_size, num_channel, height, width)
            ss = label[...,self.a]
            Ss.append(ss)
            for s in range(2):
                Xs[s].append(x[ss==s])
                b_local_group[s] += (ss==s).sum()
        for s in range(2):
            Xs[s] = torch.cat(Xs[s]).to(self.device)  # (totalsize_s, num_channel, dim)
        X = torch.cat(Xs)  # (totalsize, num_channel, dim)
        S = torch.cat(Ss).float().unsqueeze(-1).to(self.device)
        X -= X.mean(0, keepdim=True) 
        S -= S.mean() # (totalsize, 1)
        if constraint in ['mean', 'all']:
            print("mean matching...")
            # Nullspace
            XS = torch.einsum("tcd,te->cde", X, S) # (num_channel, dim, 1)
            
            # U, _ = torch.linalg.qr(XS.cpu(), mode='complete')
            # R = U[...,1:].to(self.device)  # (num_channel, dim, dim-1)
            # XR = torch.einsum("tcd,cde->tce", X, R)
            # RXXR = torch.einsum("tce,tcf->cef", XR, XR)  # (num_channel, dim-1, dim-1)
            
            R = []
            for c in range(self.num_channel):
                R.append(torch.from_numpy(null_space(XS[c].mT.numpy())))
            R = torch.stack(R)
            XR = torch.einsum("tcd,cde->tce", X, R)
            RXXR = torch.einsum("tce,tcf->cef", XR, XR)  # (num_channel, dim-1, dim-1)
            
            if constraint == 'mean':
                Sigma = RXXR
            else:
                print("covariance matching...")
                # Group-conditional covariances
                Sigmas = [None, None]
                for s in range(2):
                    Sigmas[s] = torch.einsum("tcd,tce->cde", Xs[s], Xs[s]) / Xs[s].size(0)
                RSR = torch.einsum("ced,cdf->cef", torch.einsum("cde,cdf->cef", R, Sigmas[0]-Sigmas[1]), R)
                eigvals, eigvecs = torch.linalg.eigh(RSR.cpu())
                singvals = torch.abs(eigvals) # (num_channel, dim-1)
                singvals_argsort_top = torch.argsort(singvals, dim=1, descending=True)[...,target_unfair_dim:].squeeze()
                Q = eigvecs[...,singvals_argsort_top].to(self.device) # (num_channel, dim-1, dim-1-m)
                Sigma = torch.einsum("cme,cen->cmn", torch.einsum("cem,cef->cmf", Q, RXXR), Q) # (num_channel, dim-1-m, dim-1-m)
        else:
            Sigma = torch.einsum("tcd,tce->cde", X,X) / X.size(0)
        print("PCA...")
        print("Covariance Size:", Sigma.size())
        _, eigvecs = np.linalg.eigh(Sigma.cpu().numpy())
        self.V = torch.from_numpy(eigvecs[...,-target_pca_dim:]).to(self.device)
        # eigvals, eigvecs = torch.linalg.eigh(Sigma.cpu()) # eigenvalues in ascending order
        # self.V = eigvecs[...,-target_pca_dim:].to(self.device) # (num_channel, *, k)
        if constraint == 'mean':
            self.V = torch.einsum("cde,cek->cdk", R, self.V)
        elif constraint == 'all':
            self.V = torch.einsum("cde,cek->cdk", R, torch.einsum("cem,cmk->cek", Q, self.V))