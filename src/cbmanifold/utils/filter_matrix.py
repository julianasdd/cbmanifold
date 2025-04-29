import numpy as np
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from scipy.signal import filtfilt
from scipy.ndimage import gaussian_filter


def filter_matrix_old(X, sigma=1.5, L=-1, scheme=0):
    """
    Filter a matrix of spike trains with a Gaussian kernel.

    Parameters:
        X: numpy.ndarray
            Binary matrix where each column is a spike train (Time x Trials).
        sigma: float, optional
            Standard deviation of the Gaussian kernel (default: 1.5 ms).
        L: int, optional
            Length of the Gaussian kernel (default: -1, same as the data).
        scheme: int, optional
            0 for forward-backward filtering, 1 for standard filtering (default: 0).

    Returns:
        numpy.ndarray
            Filtered spike train matrix.
    """
    if sigma <= 0:
        return X

    # Calculate L if it's set to -1
    if L == -1:
        L = 2 * int(np.ceil(sigma * 3)) + 1

    # Create the Gaussian kernel
    ker = gaussian(L, L / 2 / sigma)
    ker = ker / np.sum(ker)

    nker = len(ker)

    # Add padding to the input matrix X
    B = np.zeros((nker, X.shape[1]))
    Y = np.vstack([B, X, B])

    padlen = L * 2  # max(L, 2 * X.shape[0])

    if scheme > 0:
        # Standard filtering
        # Xf = np.zeros_like(X)
        Xf = X
        for i in range(X.shape[1]):
            Xf[:, i] = convolve(Y[:, i], ker, mode="same")
    else:
        # Forward-backward filtering
        Xf = filtfilt(ker, 1, Y, axis=0, padlen=padlen)

    # Remove padding
    return Xf[nker : nker + X.shape[0], :]


def filter_matrix(X, sigma=1.5, L=-1):
    """
    Filter a matrix of spike trains with a Gaussian kernel.

    Parameters:
        X: numpy.ndarray
            Binary matrix where each column is a spike train (Time x Trials).
        sigma: float, optional
            Standard deviation of the Gaussian kernel (default: 1.5 ms).
        L: int, optional
            Length of the Gaussian kernel (default: -1, same as the data).

    Returns:
        numpy.ndarray
            Filtered spike train matrix.
    """
    if sigma <= 0:
        return X

    Xf = gaussian_filter(X, sigma, axes=[0])

    # Remove padding
    return Xf
