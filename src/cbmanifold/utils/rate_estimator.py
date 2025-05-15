import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import get_window


def get_rate_matrix(tspike, tbounds, method = "fracrate", **options):

    tend = np.ceil(tspike.max()+1).astype(int)

    rate_series = eval(method)(tspike, tend, **options)
    ntrials = tbounds.shape[0]
    rates = np.zeros((ntrials, 1800))

    for i in range(ntrials):
        tbegin_end = tbounds[i, :]
        trial_len = int((tbounds[i, 1] - tbounds[i, 0] + 1))
        curr_trial = rate_series[int(tbegin_end[0]) : int(tbegin_end[1] + 1)]
        rates[i, :curr_trial.shape[0]] = curr_trial

    return rates

# rate matrix function to use fractioned rate
def fracrate_matrix(spike_matrix, wsize=15, window_type="tukey"):
    """
    Computes firing rates from a binary spike matrix (trials x time).
    
    Parameters:
    - spike_matrix: np.ndarray of shape (n_trials, time)
    - wsize: int, window size for smoothing
    - window_type: str, type of window to use (e.g., "tukey", "hann")
    
    Returns:
    - rate_matrix: np.ndarray of same shape as spike_matrix
    """
    n_trials, L = spike_matrix.shape
    rate_matrix = np.zeros_like(spike_matrix, dtype=float)

    win = get_window(window_type, wsize)
    win /= win.sum()

    for i in range(n_trials):
        tspike = np.where(spike_matrix[i] == 1)[0]
        r = np.zeros(L)
        if len(tspike) >= 2:
            isi = np.diff(tspike) / 1e3  # convert to seconds
            for j in range(len(tspike) - 1):
                r[tspike[j]:tspike[j+1]] = 1 / isi[j]

        # Smooth the rate using the convolution
        rx = np.convolve(r, win, mode="valid")
        delta = L - rx.size
        d2 = delta // 2 + 1
        r[:d2] = rx[0]
        r[d2:d2+rx.size] = rx
        r[d2+rx.size:] = rx[-1]

        rate_matrix[i] = r

    return rate_matrix


def fracrate(tspike, tend, wsize=5, window_type="tukey"):
    L = tend + 1

    r = np.zeros(L)
    isi = np.diff(tspike) / 1e3
    ispike = np.round(tspike).astype(int)
    for i in range(ispike.size - 1):
        ibeg = ispike[i]
        iend = ispike[i + 1]
        r[ibeg:iend] = 1 / isi[i]

    win = get_window(window_type, wsize)
    win /= win.sum()
    rx = np.convolve(win, r, mode="valid")

    delta = r.size - rx.size
    d2 = delta // 2 + 1
    r[:d2] = rx[0]
    r[d2 : (rx.size + d2)] = rx
    r[(rx.size + d2) :] = rx[-1]

    return r


def gaussian_filter(tspike, tend, wsize=5):

    tt = np.arange(0, tend + 2)
    n, _ = np.histogram(tspike, tt)
    r = gaussian_filter1d(n.astype("double"), wsize)*1e3

    return r
