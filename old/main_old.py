from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from multiprocessing import Pool
import numpy as np
from tqdm.auto import tqdm, trange
from typing import Iterable, Optional
from utilities import Problem
import warnings


def oja_subgradient(problem, V0, rho, N, alpha0=0.01, B0=5, B1=5):
    d, k = problem.d, problem.k

    if V0 is None:
        V = np.random.rand(d, k)
        V, _ = np.linalg.qr(V)
    else:
        V = V0.copy()

    alpha = alpha0
    history = [V]
    block_sizes = []
    for n in tqdm(range(1, N + 1)):
        #if n % (N//10) == 0:
        #    alpha *= 0.5
        G0, G1 = np.zeros((d, k)), np.zeros((d, k))
        q0, q1 = 0, 0
        m0, m1 = 0, 0
        l = 0
        while (m0 < B0 or m1 < B1):# and (m0+m1)<2*(B0+B1):
            s, x = problem.sample()
            if s == 0:
                G0 = G0 + np.matmul(x, np.matmul(x.T, V))
                m0 += 1
                q0 += np.inner(x.reshape(-1), x.reshape(-1))
            else:
                G1 = G1 + np.matmul(x, np.matmul(x.T, V))
                m1 += 1
                q1 += np.inner(x.reshape(-1), x.reshape(-1))
            l += 1
        block_sizes.append(l)
        G = (1 / m1) * G1 + (1 / m0) * G0
        tilde_G = (1 / m1) * G1 - (1 / m0) * G0
        q = (1 / m1) * q1 - (1 / m0) * q0
        s = np.sign(q - np.trace(V.T @ tilde_G))

        V, _ = np.linalg.qr(V + alpha * (G + rho * s * tilde_G))
        history.append(V)
    return (history, block_sizes)

def _proximal_ascent(problem, rho, gamma, tilde_V, J, B0, B1, n0, n1, q0, q1):
    d, k = problem.d, problem.k
    tilde_Gs = [tilde_V.copy()]
    l = 0
    for j in range(1, J+1):
        G0, G1 = np.zeros((d, k)), np.zeros((d, k))
        m0, m1 = 0, 0
        while m0 < B0 or m1 < B1:
            s, x = problem.sample()
            if s == 0:
                G0 = G0 + np.matmul(x, np.matmul(x.T, tilde_Gs[-1]))
                m0 += 1
                q0 += np.inner(x.reshape(-1), x.reshape(-1))
            else:
                G1 = G1 + np.matmul(x, np.matmul(x.T, tilde_Gs[-1]))
                m1 += 1
                q1 += np.inner(x.reshape(-1), x.reshape(-1))
            l += 1
        tilde_Gs.append((1 / m1) * G1 - (1/m0) * G0)
    n0, n1 = n0+m0, n1+m1
    q = q1 / n1 - q0 / n0
    sgn = np.sign(q - np.trace(tilde_V.T @ tilde_Gs[1]))

    P = np.zeros_like(tilde_V)
    for j in range(J+1):
        P += (rho * gamma * sgn)**j * tilde_Gs[j]
    
    V,_ = np.linalg.qr(P)
    return V, l, n0, n1, q0, q1


def proximal_block_method(problem, V0, rho, N, J=2, alpha0=0.01, gamma=0.1, gamma_min=0, B0=5, B1=5, alg='oja', _proximal_ascent=_proximal_ascent):
    assert alg in ['oja', 'NPM']
    d, k = problem.d, problem.k

    if V0 is None:
        V = np.random.rand(d, k)
        V, _ = np.linalg.qr(V)
    else:
        V = V0.copy()

    alpha = alpha0
    history = [V]
    block_sizes = []
    n0, n1 = 0, 0
    q0, q1 = 0, 0
    for n in tqdm(range(1, N + 1)):
        #if n % (N//10) == 0:
        #    alpha *= 0.5
        G0, G1 = np.zeros((d, k)), np.zeros((d, k))
        #n0, n1 = 0, 0
        #q0, q1 = 0, 0
        m0, m1 = 0, 0
        l = 0
        while m0 < B0 or m1 < B1:
            s, x = problem.sample()
            if s == 0:
                G0 = G0 + np.matmul(x, np.matmul(x.T, V))
                m0 += 1
                q0 += np.inner(x.reshape(-1), x.reshape(-1))
            else:
                G1 = G1 + np.matmul(x, np.matmul(x.T, V))
                m1 += 1
                q1 += np.inner(x.reshape(-1), x.reshape(-1))
            l += 1
        G = (1 / m1) * G1 + (1 / m0) * G0
        n0, n1 = n0+m0, n1+m1

        if alg == 'oja':   tilde_V = V + alpha * G
        elif alg == 'NPM': tilde_V = G
        else:              raise NotImplementedError
        
        #if n%(N//10) == 0:
        #    gamma = max(gamma * 0.5, gamma_min)
        V, additional_block_size, n0, n1, q0, q1 = _proximal_ascent(problem, rho, gamma, tilde_V, J, B0, B1, n0, n1, q0, q1)
        history.append(V)
        block_sizes.append(l + additional_block_size)
    return (history, block_sizes)


def plot_all(
        problem: Problem,
        rhos: list,
        history: Iterable,
        name: str,
        show: bool = True,
        block_size: Optional[list]=None,
        legend_cols: Optional[int]=1
    ):
    plt.rc('legend',fontsize=5)
    num_ls = len(rhos)
    colors = pl.cm.jet(np.linspace(0,1,num_ls))
    xscale='linear' # 'log' or 'linear'
    legend_kwargs={
        'ncols': legend_cols,
        'bbox_to_anchor':(1.04, 0.5),
        'loc': "center left",
    }
    save_kwargs={
        'dpi': 1200,
        'bbox_inches': "tight"
    }
    algo_name = name.split('_')[0].capitalize()

    # explained variance suboptimality gap
    plt.figure()
    plt.title(f"Ratio of explained variance ({algo_name})")
    plt.ylim(0.9, 1.005)
    for i in trange(num_ls):
        exp_var = [np.trace(problem.Sigma @ V @ V.T) / problem.total_var for V in history[i]]
        plt.plot(exp_var, alpha=0.9, color=colors[i], label=str(rhos[i]))
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], **legend_kwargs)
    plt.xlabel("iteration")
    plt.xscale(xscale)
    plt.savefig(f"expvar_{name.lower()}.pdf", **save_kwargs)
    if show: plt.show()

    # difference in reconstruction error
    plt.figure()
    plt.title(f"Fairness measure ({algo_name})")
    q = np.trace(problem.Q)
    # plt.axhline(y=np.abs(np.trace(problem.Q @ problem.V_star @ problem.V_star.T) - q), color='r', linestyle='-')
    for i in trange(num_ls):
        fairness_Q = [np.abs(np.trace(problem.Q @ V @ V.T) - q) for V in history[i]]
        plt.plot(fairness_Q, alpha=0.9, color=colors[i], label=str(rhos[i]))
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], **legend_kwargs)
    plt.xlabel("iteration")
    plt.xscale(xscale)
    plt.savefig(f"fairness_Q_{name.lower()}.pdf", **save_kwargs)
    if show: plt.show()

    # loss function
    plt.figure()
    plt.title(f"Loss function ({algo_name})")
    for i in trange(num_ls):
        loss = np.array(
            [np.trace(problem.Sigma @ V @ V.T) - rhos[i] * np.abs(np.trace(problem.Q @ V @ V.T) - q) for V in history[i]])
        plt.plot(loss, alpha=0.9, color=colors[i], label=str(rhos[i]))
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], **legend_kwargs)
    plt.xlabel("iteration")
    plt.xscale(xscale)
    plt.savefig(f"losses_{name.lower()}.pdf", **save_kwargs)
    if show: plt.show()

    # block sizes
    if block_size is not None:
        plt.figure()
        plt.title(f"Block sizes ({algo_name})")
        for i in trange(num_ls):
            plt.plot(block_size[i], alpha=0.9, color=colors[i], label=str(rhos[i]))
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], **legend_kwargs)
        plt.savefig(f"block_sizes_{name.lower()}.pdf", **save_kwargs)
        if show: plt.show()
    
    plt.close('all')


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    d, k_, p = 10, 3, 0.4  # k_: true k
    N = int(1e6)
    eps = 0.25  # larger: smaller divergent rho / higher lower-bd of expvar (smaller gap)
    seed = 42
    rng = np.random.default_rng(seed) # fixed random generator
    
    tmp = rng.normal(size=(d, k_))
    tmp = tmp / np.linalg.norm(tmp)
    Sigma = tmp @ tmp.T + 2*eps * np.eye(d)
    w, v = np.linalg.eigh(Sigma)

    tmp = v[:, :-k_]
    D = rng.random(d-k_)
    D = (D - D.min()) / (D.max()-D.min())  # D.min() ==> 0, D.max() ==> 1
    D = (2*D-1)*eps                        # D.min() ==> -eps, D.max() ==> eps
    print(D)
    Q = tmp @ np.diag(D) @ tmp.T

    Sigma0 = 0.5 * (Sigma - Q)
    Sigma1 = 0.5 * (Sigma + Q)

    abs_tr_list = []
    for i in range(10000):
        V = np.random.randn(d, k_)
        V,_ = np.linalg.qr(V)
        tr = np.trace((np.eye(d)-np.matmul(V, V.T)) @ Q)
        abs_tr_list.append(abs(tr))
    
    # plot eigenvalue distribution?
    w_Sigma, _ = np.linalg.eigh(Sigma)
    plt.figure(100)
    plt.plot(w_Sigma, '-o')
    plt.title("Eigenvalue distribution of Sigma")
    plt.savefig("eig_sigma.pdf", dpi=1200)

    w_Q, _ = np.linalg.eigh(Q)
    plt.figure(101)
    plt.plot(w_Q, '-o')
    plt.title("Eigenvalue distribution of Q")
    plt.savefig("eig_Q.pdf", dpi=1200)

    plt.figure(102)
    plt.hist(abs_tr_list, bins=100)
    plt.title("histogram of fairness measure over random V")
    plt.savefig("hist_random_fairness.pdf", dpi=1200)

    #plt.show()
    plt.close('all')

    k = 3  # problem k
    problem = Problem(d, k, p, Sigma0, Sigma1)
    

    V0 = rng.random(size=(d, k))
    V0, _ = np.linalg.qr(V0)

    ## Offline : subgradient method ##
    rhos = [round(rho, 8) for rho in np.linspace(0, 2, 50 + 1)]
    rhos.reverse()
    alpha = 1e-1
    params = [(V0, rho, N, alpha) for rho in rhos]
    with Pool() as p:
        #history = p.starmap(problem.solve_offline, params)
        history = p.starmap_async(problem.solve_offline, params)
        history = list(history.get())
    plot_all(problem, rhos, history, name=f'offline_N{N}', show=False, legend_cols=2)

    ## FD-BSOM ##
    """rhos = [round(rho, 8) for rho in np.linspace(0, 5, 25 + 1)]
    rhos.reverse()
    alpha = 1e-3
    B0, B1 = 10,10
    params = [(problem, V0, rho, N, alpha, B0, B1) for rho in rhos]
    with Pool() as p:
        #output = p.starmap(oja_subgradient, params)
        output = p.starmap_async(oja_subgradient, params)
        output = list(output.get())
    history, block_size = tuple(map(list, zip(*output)))
    plot_all(problem, rhos, history, block_size=block_size, name=f'streaming_BSOM_N{N}_1', show=False, legend_cols=2)"""

    ## FD-PBM ##
    """rhos = [round(rho, 8) for rho in np.linspace(0, 2, 10 + 1)]
    rhos.reverse()
    rhos.reverse()
    alpha = 1e-3
    B0, B1 = 3,3
    J = 2           # number of inner loop
    gamma = 0.01    # proximal coeff
    gamma_min = 0   # gamma/32 #0.1 / 16
    alg = 'oja'     # algorithm parameter, ['oja', 'NPM']
    params = [(problem, V0, rho, N, J, alpha, gamma, gamma_min, B0, B1, alg) for rho in rhos]
    with Pool() as p:
        #output = p.starmap(proximal_block_method, params)
        output = p.starmap_async(proximal_block_method, params)
        output = list(output.get())
    history, block_size = tuple(map(list, zip(*output)))
    plot_all(problem, rhos, history, block_size=block_size, name=f'streaming_PBM_N{N}_inner{J}', show=False)"""
