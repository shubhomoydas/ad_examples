import numpy as np
from .utils import order, nrow, rank


#######################################################
# Computes the Area under the ROC curve for instances
# ranked by score
#
# d - n x 2 matrix where col 1 is 0/1 indicating
#     whether corresponding index is normal(0) or
#     anomalous(1) col 2 contains the scores
#     The scores should be in increasing order (lower
#     values are more anomalous).
#######################################################
# D = np.reshape(np.array([0,1,0,0,1,0,2,2,3,4,2,6], dtype=float), (6,2), order='F')
# fn_auc(D)
def fn_auc(d):
    x = d[order(d[:, 1]), :]
    # m - number of anomalies
    N = nrow(d)  # total number of instances
    m = np.sum(d[:, 0])  # number of anomalies
    n = N - m  # number of unseen nominal instances
    r = 0.0
    for i in range(N):
        if x[i, 0] == 1:
            r += n
        else:
            n -= 1.0
    auc = r / float(m * (N - m))
    return auc


#######################################################
# d - n x 2 matrix where col 1 is 0/1 indicating
#     whether corresponding index is normal(0) or
#     anomalous(1) col 2 contains the scores
#     The scores should be in increasing order (lower
#     values are more anomalous).
# k - vector of integers indicating at which positions
#     the precision will be computed.
#######################################################
# D = np.reshape(np.array([0,1,0,0,1,0,2,2,3,4,2,6], dtype=float), (6,2), order='F')
# K = np.array([10,20,50,100])
# fn_precision(D, K)
def fn_precision(d, k):
    x = d[order(d[:, 1]), :]
    y = x[:, 0]
    num_anom = np.sum(y)
    ranks = rank(x[:, 1], ties_method="average")
    c_y = np.cumsum(y)
    n = nrow(x)
    k = np.minimum(k, n)
    k = np.maximum(k, 1)
    if num_anom == 0:
        prec_k = np.zeros(len(k))  # Precision@K
        avg_prec = 0
        low_rank = 0
    else:
        prec_k = np.zeros(len(k))  # Precision@K
        rank_pos = np.zeros(len(k), dtype=int)
        for i in range(len(k)):
            temp_pos = np.where(ranks <= k[i])[0]
            if len(temp_pos) > 0:
                rank_pos[i] = np.max(temp_pos)
                prec_k[i] = c_y[rank_pos[i]] / float(k[i])
        pos = np.where(y == 1)[0]
        apr_pos = ranks[pos]
        avg_prec = np.sum(c_y[pos] / apr_pos) / float(num_anom)
        low_rank = ranks[max(pos)]
    pres = list(prec_k)
    pres.extend([avg_prec, low_rank, n, int(num_anom)])
    return pres

