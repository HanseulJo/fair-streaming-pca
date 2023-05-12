from sklearn import metrics
import torch


def mmd(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def nullspace(At: torch.Tensor, rcondt=None):
    """
    https://discuss.pytorch.org/t/nullspace-of-a-tensor/69980/4
    """
    device = At.device
    ut, st, vht = torch.linalg.svd(At.cpu(), some=False, compute_uv=True)
    vht=torch.transpose(vht, -2,-1)     
    Mt, Nt = ut.shape[-2], vht.shape[-1] 
    if rcondt is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcondt
    numt= torch.sum(st > tolt, dtype=int)
    nullspace = vht[...,numt:,:].T.cpu().conj()
    # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
    return nullspace.to(device)