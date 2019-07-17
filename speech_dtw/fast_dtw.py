"""Fast dynamic time warping (DTW) accelerated by Numba JIT compiler.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: June 2019

Based on implementation from https://en.wikipedia.org/wiki/Dynamic_time_warping
and https://github.com/kamperh/speech_dtw.

Some really great slides on DTW https://www.cs.unm.edu/~mueen/DTW.pdf.

Test performance with timeit:

- `dtw_cdist` warping between two timeseries matrices (see `_test_data_matrix`)

```
    python  -m timeit \
        -s "import fast_dtw as dtw" \
        -s "X_a, X_b = dtw._test_data_matrix()" \
        "dtw.dtw_cdist(X_a, X_b, 'cosine')"
```

- `dtw_dist_cosine` warping between two timeseries vectors (see `_test_data_vector`)

```
    python  -m timeit \
        -s "import fast_dtw as dtw" \
        -s "x_a, x_b = dtw._test_data_vector()" \
        -s "cost_mat = dtw.dtw_dist_cosine(x_a, x_b)" \
        -s "print('cost value:', dtw.get_final_cost(cost_mat))" \
        -s "print('cost path:', dtw.get_cost_path(cost_mat))" \
        "dtw.dtw_dist_cosine(x_a, x_b)"
```
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numba
import numpy as np


from sklearn.preprocessing import StandardScaler


@numba.jit(nopython=True)
def fast_cosine(v_a, v_b):
    """Pure python cosine distance faster than NumPy implementation with Numba."""
    n = v_a.shape[0]  # should match len(v_b)
    dot = 0.
    norm_v_a = 0.
    norm_v_b = 0.
    for i in range(n):
        dot += v_a[i] * v_b[i]
        norm_v_a += v_a[i] * v_a[i]
        norm_v_b += v_b[i] * v_b[i]
    return 1. - dot / ((norm_v_a * norm_v_b) ** 0.5)


@numba.jit(nopython=True)
def dtw_dist_cosine(x_a, x_b):
    """Compute dynamic time warping (DTW) cosine distance between two temporal sequences."""
    m_a = x_a.shape[0]
    m_b = x_b.shape[0]
    cost_mat = np.full((m_a + 1, m_b + 1), np.inf)  # init. (m_a+1) x (m_b+1) "inf" cost matrix
    cost_mat[0, 0] = 0.  # set cost at alignment t=(0,0) to zero

    for i in range(m_a):
        for j in range(m_b):
            cost = fast_cosine(x_a[i], x_b[j])
            cost_mat[i + 1, j + 1] = cost + min(cost_mat[i, j + 1],  # insertion
                                                cost_mat[i + 1, j],  # deletion
                                                cost_mat[i, j])  # match
    return cost_mat


@numba.jit(nopython=True)
def fast_euclidean_squared(v_a, v_b):
    """Pure python euclidean distance faster than NumPy implementation with Numba."""
    n = v_a.shape[0]  # should match len(v_b)
    sum_square_diffs = 0.
    for i in range(n):
        sum_square_diffs += (v_a[i] - v_b[i]) ** 2
    return sum_square_diffs


@numba.jit(nopython=True)
def dtw_dist_euclidean(x_a, x_b, return_paths=False):
    """Compute dynamic time warping (DTW) euclidean distance between two temporal sequences."""
    m_a = x_a.shape[0]
    m_b = x_b.shape[0]
    cost_mat = np.full((m_a + 1, m_b + 1), np.inf)  # init. (m_a+1) x (m_b+1) "inf" cost matrix
    cost_mat[0, 0] = 0.  # set cost at alignment t=(0,0) to zero

    for i in range(m_a):
        for j in range(m_b):
            cost = fast_euclidean_squared(x_a[i], x_b[j]) ** 0.5
            cost_mat[i + 1, j + 1] = cost + min(cost_mat[i, j + 1],  # insertion
                                                cost_mat[i + 1, j],  # deletion
                                                cost_mat[i, j])  # match
    return cost_mat


@numba.jit(nopython=True)
def dtw_dist_euclidean_squared(x_a, x_b):
    """Compute dynamic time warping (DTW) squared euclidean distance between two temporal sequences."""
    m_a = x_a.shape[0]
    m_b = x_b.shape[0]
    cost_mat = np.full((m_a + 1, m_b + 1), np.inf)  # init. (m_a+1) x (m_b+1) "inf" cost matrix
    cost_mat[0, 0] = 0.  # set cost at alignment t=(0,0) to zero

    for i in range(m_a):
        for j in range(m_b):
            cost = fast_euclidean_squared(x_a[i], x_b[j])
            cost_mat[i + 1, j + 1] = cost + min(cost_mat[i, j + 1],  # insertion
                                                cost_mat[i + 1, j],  # deletion
                                                cost_mat[i, j])  # match
    return cost_mat


@numba.jit(nopython=True)
def get_final_cost(cost_mat):
    return cost_mat[-1, -1]


@numba.jit(nopython=True)
def get_cost_path(cost_mat):
    i, j = cost_mat.shape
    i = i - 2
    j = j - 2
    path = [(i, j)]
    while i > 0 or j > 0:
        traceback = np.argmin(
            np.array([
                cost_mat[i, j],
                cost_mat[i, j + 1],
                cost_mat[i + 1, j]]))
        if traceback == 0:  # match
            i = i - 1
            j = j - 1
        elif traceback == 1:  # insertion
            i = i - 1
        elif traceback == 2:  # deletion
            j = j - 1
        path.append((i, j))
    return path


def dtw_cdist(X_a, X_b, metric="cosine"):
    """Compute distance between each pair in two collections of input matrices.
    
    Matrix `X_a` has shape `[m_a, t_a_i , n]`, with `m_a` observations, each
    with `t_a_i` observation timesteps in n-dimensional space. Similar for
    matrix `X_b`. Observations may have variable number of timesteps.
    
    Output has shape [`m_a`, `m_b`], where each entry `[i, j]` is the warped
    path distance `dtw_dist(X_a[i], X_b[j])` according to the specified metric. 
    """
    if callable(metric):
        dtw_dist = metric
    else:
        if metric == "cosine":
            dtw_dist = dtw_dist_cosine
        elif metric == "euclidean":
            dtw_dist = dtw_dist_euclidean
        elif metric == "euclidean_squared":
            dtw_dist = dtw_dist_euclidean_squared
        else:
            raise ValueError(
                "Metric not in list of allowed values {} Got: {}".format(
                    ["cosine", "euclidean", "euclidean_squared"],
                    metric))
    return _fast_dtw_cdist(X_a, X_b, dtw_dist)


@numba.jit(nopython=True, parallel=True, nogil=True)
def _fast_dtw_cdist(X_a, X_b, dtw_dist):
    m_a = X_a.shape[0]
    m_b = X_b.shape[0]
    cdist = np.empty((m_a, m_b), np.float64)
    for i in numba.prange(m_a):
        for j in numba.prange(m_b):
            cost_mat = dtw_dist(X_a[i], X_b[j])
            cdist[i, j] = get_final_cost(cost_mat)  # / len(get_cost_path(cost_mat))
    return cdist


def _test_data_matrix():
    np.random.seed(42)
    X_a = np.random.rand(10, 110, 40)
    X_b = np.random.rand(10, 110, 40)
    return X_a, X_b


def _test_data_vector():
    np.random.seed(42)
    X_a = np.random.rand(110, 40)
    X_b = np.random.rand(110, 40)
    return X_a, X_b
