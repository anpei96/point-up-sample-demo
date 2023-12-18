from __future__ import division
import numpy as np
import logging

from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds

def _my_svd(M, k, algorithm):
    if algorithm == 'randomized':
        (U, S, V) = randomized_svd(
            M, n_components=min(k, M.shape[1]-1), n_oversamples=20)
    elif algorithm == 'arpack':
        (U, S, V) = svds(M, k=min(k, min(M.shape)-1))
        S = S[::-1]
        U, V = svd_flip(U[:, ::-1], V[::-1])
    else:
        raise ValueError("unknown algorithm")
    return (U, S, V)

def svt_solve(
        A, 
        mask, 
        tau=None, 
        delta=None, 
        epsilon=1e-2,
        rel_improvement=-0.01,
        max_iterations=1000,
        algorithm='arpack'):
    """
    Solve using iterative singular value thresholding.

    [ Cai, Candes, and Shen 2010 ]

    Parameters:
    -----------
    A : m x n array
        matrix to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    tau : float
        singular value thresholding amount;, default to 5 * (m + n) / 2

    delta : float
        step size per iteration; default to 1.2 times the undersampling ratio

    epsilon : float
        convergence condition on the relative reconstruction error

    max_iterations: int
        hard limit on maximum number of iterations

    algorithm: str, 'arpack' or 'randomized' (default='arpack')
        SVD solver to use. Either 'arpack' for the ARPACK wrapper in 
        SciPy (scipy.sparse.linalg.svds), or 'randomized' for the 
        randomized algorithm due to Halko (2009).

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    logger = logging.getLogger(__name__)
    if algorithm not in ['randomized', 'arpack']:
        raise ValueError("unknown algorithm %r" % algorithm)
    Y = np.zeros_like(A)

    if not tau:
        tau = 5 * np.sum(A.shape) / 2
    if not delta:
        delta = 1.2 * np.prod(A.shape) / np.sum(mask)

    r_previous = 0

    for k in range(max_iterations):
        if k == 0:
            X = np.zeros_like(A)
        else:
            sk = r_previous + 1
            (U, S, V) = _my_svd(Y, sk, algorithm)
            while np.min(S) >= tau:
                sk = sk + 5
                (U, S, V) = _my_svd(Y, sk, algorithm)
            shrink_S = np.maximum(S - tau, 0)
            r_previous = np.count_nonzero(shrink_S)
            diag_shrink_S = np.diag(shrink_S)
            X = np.linalg.multi_dot([U, diag_shrink_S, V])
        Y += delta * mask * (A - X)

        recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        # print("tmp-0: ", A)
        # print("tmp-1: ", np.linalg.norm(mask * (X - A)))
        # print("tmp-2: ", np.linalg.norm(mask * A))
        # return 
        
        if k % 1 == 0:
            logger.info("Iteration: %i; Rel error: %.4f" % (k + 1, recon_error))
        if recon_error < epsilon:
            break

    return X

def pmf_solve(A, mask, k, mu, epsilon=1e-3, max_iterations=100):
    """
    Solve probabilistic matrix factorization using alternating least squares.

    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.

    [ Salakhutdinov and Mnih 2008 ]
    [ Hu, Koren, and Volinksy 2009 ]

    Parameters:
    -----------
    A : m x n array
        matrix to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    k : integer
        how many factors to use

    mu : float
        hyper-parameter penalizing norm of factored U, V

    epsilon : float
        convergence condition on the difference between iterative results

    max_iterations: int
        hard limit on maximum number of iterations

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    logger = logging.getLogger(__name__)
    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)

    for _ in range(max_iterations):

        for i in range(m):
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([V.T, C_u[i], A[i,:]]))

        for j in range(n):
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([U.T, C_v[j], A[:,j]]))

        X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        if _ % 1 == 0:
            logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if mean_diff < epsilon:
            break
        prev_X = X

    return X

def biased_mf_solve(A, mask, k, mu, epsilon=1e-3, max_iterations=100):
    """
    Solve biased probabilistic matrix factorization via alternating least
    squares.

    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.

    [ Paterek 2007 ]
    [ Koren, Bell, and Volinksy 2009 ]

    Parameters:
    -----------
    A : m x n array
        matrix to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    k : integer
        how many factors to use

    mu : float
        hyper-parameter penalizing norm of factored U, V and biases beta, gamma

    epsilon : float
        convergence condition on the difference between iterative results

    max_iterations: int
        hard limit on maximum number of iterations

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    logger = logging.getLogger(__name__)
    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    beta = np.random.randn(m)
    gamma = np.random.randn(n)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T) + \
                     np.outer(beta, np.ones(n)) + \
                     np.outer(np.ones(m), gamma)

    for _ in range(max_iterations):

        # iteration for U
        A_tilde = A - np.outer(np.ones(m), gamma)
        V_tilde = np.c_[np.ones(n), V]
        for i in range(m):
            U_tilde = np.linalg.solve(np.linalg.multi_dot([V_tilde.T, C_u[i],
                                                           V_tilde]) +
                                      mu * np.eye(k + 1),
                                      np.linalg.multi_dot([V_tilde.T, C_u[i],
                                                           A_tilde[i,:]]))
            beta[i] = U_tilde[0]
            U[i] = U_tilde[1:]

        # iteration for V
        A_tilde = A - np.outer(beta, np.ones(n))
        U_tilde = np.c_[np.ones(m), U]
        for j in range(n):
            V_tilde = np.linalg.solve(np.linalg.multi_dot([U_tilde.T, C_v[j],
                                                           U_tilde]) +
                                                           mu * np.eye(k + 1),
                                      np.linalg.multi_dot([U_tilde.T, C_v[j],
                                                           A_tilde[:,j]]))
            gamma[j] = V_tilde[0]
            V[j] = V_tilde[1:]

        X = np.dot(U, V.T) + \
            np.outer(beta, np.ones(n)) + \
            np.outer(np.ones(m), gamma)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        if _ % 1 == 0:
            logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if mean_diff < epsilon:
            break
        prev_X = X

    return X
