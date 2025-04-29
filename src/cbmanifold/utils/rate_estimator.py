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
