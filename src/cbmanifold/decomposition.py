import numpy as np
from scipy.linalg import eigh
import warnings

from . import linear_model


def pmPCA(x, normalize=True):
    """Poor man's PCA analysis. This is a vanilla implementation of principal component
    analysis to make sure no preprocessing of data. The input is assumed to be a
    Nsamples x dim matrix.

    Args:
        x: Input data matrix (Nsamples x dim)
        normalize: Whether to normalize the eigenvectors by sqrt(Nsamples)

    Returns:
        d: Covariance eigenvalues
        p: Principal components
        v: Eigenvectors

    Written by Sungho Hong, CNS unit, OIST
    March 2023
    """

    # Find non-NaN rows
    ii = ~np.isnan(np.mean(x, axis=1))
    y = x[ii, :]

    # Compute covariance matrix
    cc = np.cov(y)
    assert cc.shape == (x.shape[0], x.shape[0])

    # Get eigenvalues and eigenvectors
    d, vy = eigh(cc)
    d = d[::-1]  # Sort in descending order
    vy = vy[:, ::-1]
    if normalize:
        vy = vy / np.sqrt(x.shape[0])

    # Initialize output eigenvector matrix
    nrow = x.shape[0]
    v = np.zeros((nrow, np.sum(ii)))
    v[ii, :] = vy

    # Compute principal components
    p = y.T @ vy

    return d, p, v


def reduce_dimensionality(lm, dim_target, normalize=True):
    """Reduce the dimensionality of a parametric linear model

    Args:
        lm: Parametric linear model with fields rate and drate that are NxT matrices where
            N is the number of neurons and T is the number of time points.
        dim_target: number of dimensions of the reduced model.
        normalize: Whether to normalize the eigenvectors by sqrt(Nsamples)
    Returns:
        lm_reduced: reduced model with fields p, dp, etc. p and dp are nmode x T matrices.
    """

    _, L = lm.rate.shape

    # PCA on the independent component
    d0, _, v = pmPCA(lm.rate, normalize=False)

    # select only the valid data
    isnanrate = np.isnan(np.mean(lm.rate, axis=1))
    if np.any(isnanrate):
        warnings.warn("Found NaN values in rate... excluding them")

    nnrate = lm.rate[~isnanrate, :]
    nndrate = lm.drate[~isnanrate, :]
    v = v[~isnanrate, :]

    N = nnrate.shape[0]

    # dimensionality reduction of the independent component
    p = v[:, :dim_target].T @ nnrate
    p = p[:dim_target, :]

    # direct projection part
    dp0 = v[:, :dim_target].T @ nndrate
    dp0 = dp0[:dim_target, :]

    # copy the data and save the dimensionality reduction
    lm_reduced = linear_model.LinearModel(label=lm.label, is_dim_reduced=True)
    lm_reduced.dim = dim_target
    lm_reduced.n_params = lm.n_params
    lm_reduced.params0 = lm.params0
    lm_reduced.p = p
    lm_reduced.dp0 = dp0

    # approximate the individual rates by the dimensionality reduced representation
    lm_reduced.rate = v[:, :dim_target] @ lm_reduced.p
    lm_reduced.drate0 = v[:, :dim_target] @ lm_reduced.dp0

    # here is the core of this code to compute how PCA results are affected by small perturbations
    # begin computing perturbations
    z = nnrate.T
    zc = z - np.mean(z)

    # perturbed part
    dzc = lm.drate[~isnanrate, :].T

    # perturbed covariance matrix
    cc1 = (dzc.T @ zc + zc.T @ dzc) / L

    # prepare eigenvalue perturbation and a mixing matrix
    dl = np.zeros_like(d0)
    ckn = np.zeros((N, N))
    assert N == len(d0)

    # compute the mixing matrix
    w0 = v
    for i in range(N):
        dl[i] = w0[:, i].T @ cc1 @ w0[:, i]
        xx = np.setdiff1d(np.arange(len(d0)), i)
        ckn[xx, i] = w0[:, xx].T @ cc1 @ w0[:, i]
        ckn[xx, i] = ckn[xx, i] / (d0[i] - d0[xx])

    # eigenvector perturbation
    de = w0 @ ckn

    # indirect projection part
    lm_reduced.w0 = v[:, :dim_target]
    lm_reduced.dp1 = de[:, :dim_target].T @ (nnrate - v[:, :dim_target] @ p)

    # total perturbation = direct + indirect
    lm_reduced.dp = lm_reduced.dp0 + lm_reduced.dp1

    # approximation of the individual rate by dp1
    lm_reduced.drate1 = v[:, :dim_target] @ lm_reduced.dp1

    # full approximation including de
    lm_reduced.drate2 = (
        -v[:, dim_target:] @ de[:, dim_target:].T @ v[:, :dim_target] @ p
    )

    if normalize:
        lm_reduced.p = lm_reduced.p / np.sqrt(N)
        lm_reduced.dp = lm_reduced.dp / np.sqrt(N)
        lm_reduced.dp0 = lm_reduced.dp0 / np.sqrt(N)
        lm_reduced.dp1 = lm_reduced.dp1 / np.sqrt(N)
        lm_reduced.w0 = lm_reduced.w0 / np.sqrt(N)

    return lm_reduced
