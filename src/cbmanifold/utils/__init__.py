def flip_sign(lm, dims_to_flip_sign):
    for i in dims_to_flip_sign:
        lm.p[i, :] = -lm.p[i, :]
        lm.dp[i, :] = -lm.dp[i, :]
        lm.dp0[i, :] = -lm.dp0[i, :]
        lm.dp1[i, :] = -lm.dp1[i, :]
        lm.w0[:, i] = -lm.w0[:, i]

    return lm


# ## Data processing utils
from .filter_matrix import filter_matrix, filter_matrix_old
from .rate_estimator import *
