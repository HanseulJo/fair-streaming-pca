import matplotlib.pyplot as plt
import jax.numpy as jnp
from math import log10
from fairPCA import StreamingFairBlockPCA
import multiprocessing
import time
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from copy import deepcopy

def run_fairPCA(param):
    i, block_size, Problem, k, m = param    
    Problem.train(
        target_dim=k,
        rank=m,
        n_iter=25,
        n_iter_inner=25,
        batch_size_subspace=block_size,
        batch_size_pca=block_size,
        constraint='all',
        subspace_optimization='npm',
        pca_optimization='npm',
        seed=i,
        verbose=True,
        use_true_N=False,
        use_opt_V=True
    )
    exp_var_eps = Problem.exp_var_gt/Problem.total_var - Problem.buffer.explained_variance_ratio[-1]
    fairness_eps = jnp.linalg.norm(Problem.true_N.T @ Problem.V, 2)
    return exp_var_eps, fairness_eps



if __name__ == '__main__':
    d, k, m = 30, 3, 3
    Problem = StreamingFairBlockPCA(
        data_dim=d,
        probability=0.5,
        rank=m,  # effective rank of Sigma_gap
        eps=0.3,
        mu_scale=2,
        max_cov_eig0=4,
        max_cov_eig1=2,
        seed=2023
    )

    MAX_PROC = 10
    delta_inv = 10
    mul = 10
    # bs_exp_list = jnp.linspace(0.6, 4, 35+1)#[::-1]
    bs_exp_list = jnp.linspace(1, 4, 6+1)#[::-1]
    eps1_list = []
    eps2_list = []
    linear = LinearRegression()

    pbar = tqdm(bs_exp_list)
    for bs_exp in pbar:
        bs_exp = float(bs_exp)
        block_size = round(10 ** bs_exp)
        pbar.set_description(f'BS: 10^{bs_exp}')
        print()
        
        with multiprocessing.Pool(MAX_PROC) as p:
            result = p.map_async(run_fairPCA, [(i, block_size, deepcopy(Problem), k, m) for i in range(delta_inv*mul)])
            result = jnp.array(result.get())

        eps1_arr, eps2_arr = result.T
        
        eps1_arr, eps2_arr = eps1_arr.tolist(), eps2_arr.tolist()
        eps1_list.append(log10(sorted(eps1_arr)[-mul-1]))
        eps2_list.append(log10(sorted(eps2_arr)[-mul-1]))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].scatter(bs_exp_list[:len(eps1_list)], eps1_list, c='tab:green')
        axes[0].set_xlabel('log$_{10}$ (block size)')
        axes[0].set_ylabel('log$_{10}$ $\epsilon_1$')
        axes[0].set_title('Optimality: $\epsilon_1$', weight='bold')
        axes[1].scatter(bs_exp_list[:len(eps2_list)], eps2_list, c='tab:red')
        axes[1].set_xlabel('log$_{10}$ (block size)')
        axes[1].set_ylabel('log$_{10}$ $\epsilon_2$')
        axes[1].set_title('Fairness: $\epsilon_2$', weight='bold')
        if len(eps1_list) > 1:
            linear.fit(bs_exp_list[:len(eps1_list)].reshape(-1,1), eps1_list)
            y1 = linear.predict(bs_exp_list.reshape(-1,1))
            axes[0].plot(bs_exp_list, y1)
            axes[0].set_title(f'Optimality: $\epsilon_1$ (slope: {linear.coef_[0]:.3f})', weight='bold')
        if len(eps2_list) > 1:
            linear.fit(bs_exp_list[:len(eps2_list)].reshape(-1,1), eps2_list)
            y2 = linear.predict(bs_exp_list.reshape(-1,1))
            axes[1].plot(bs_exp_list, y2)
            axes[1].set_title(f'Fairness: $\epsilon_2$ (slope: {linear.coef_[0]:.3f})', weight='bold')
        fig.tight_layout()
        fig.savefig('PAFO_V_is_optimal.pdf')
        plt.close('all')


        


        

