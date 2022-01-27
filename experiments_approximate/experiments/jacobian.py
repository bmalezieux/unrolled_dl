import numpy as np

from pathlib import Path
from joblib import Parallel, delayed
from joblib import parallel_backend


RESULTS = Path(__file__).resolve().parents[1] / "results"
N_JOBS = 10
N_EXAMPLES = 50
RNG = np.random.default_rng(2022)


def compute_lipschitz(D):
    """
    Power iteration to compute the Lipschitz constant
    of the gradient of the data fitting term

    Parameters
    ----------
    D : np.array
        dictionary

    Returns
    -------
    float
        Lipschitz constant
    """
    iterations = 50
    u = RNG.random(D.shape[1])
    for i in range(iterations):
        u = D.T @ D @ u
        norme = np.linalg.norm(u)
        u /= norme
    return norme


def ista(y, D, n_iter, L, lambd=0.1):
    """
    ISTA

    Parameters
    ----------
    y : np.array (dim signal, number of samples)
        signal
    D : np.array (dim signal, dim sparse code)
        dictionary
    n_iter : int
        number of iterations
    L : float
        Lipschitz constant of the gradient of the data fitting term
    lambd : float, optional
        regularization hyperparameter, by default 0.1

    Returns
    -------
    (np.array(dim sparse code, number of samples), list)
        tuples containing the solution and the optimization path
    """
    out = np.zeros((D.shape[1], y.shape[1]))

    step = 1. / L
    path = [out.copy()]

    for i in range(n_iter):

        # Gradient descent
        out = out - step * D.T @ (D @ out - y)

        # Thresholding
        thresh = np.abs(out) - step * lambd
        out = np.sign(out) * np.maximum(0, thresh)

        path.append(out.copy())

    return out, path


def compute_nth_jac(n, path, row, L, D, y, K=None):
    """
    Computes nth jacobian

    Parameters
    ----------
    n : int
        number of jcobian iterates
    path : list
        optimization path from ista
    row : int
        row for which the jacobian is computed
    L : float
        Lipschitz constant of the gradient of the data fitting term
    D : np.array (dim signal, dim sparse code)
        dictionary
    y : np.array (dim signal, number of samples)
        signal
    K : int, optional
        backpropagation depth, by default None (complete backprop)

    Returns
    -------
    np.array
        Jacobian estimate
    """
    if K is None:
        K = 0
    if K > n:
        K = n
    start = n - K
    m = path[start].shape[0]
    jacobian = np.zeros((m, D.shape[1]))
    step = 1 / L
    for i in range(start + 1, n+1):
        jacobian -= step * (D[row].reshape(-1, 1) @ path[i-1].reshape(1, -1)
                            + (D[row] @ path[i-1] - y[row]) * np.eye(m)
                            + D.T @ D @ jacobian)
        support = np.zeros(m)
        support[(np.abs(path[i]) > 0).reshape(1, -1)[0]] = 1
        jacobian *= support.reshape(-1, 1)
    return jacobian


def true_jac(y, x, D):
    """
    Closed form jacobian

    Parameters
    ----------
    y : np.array
        signal
    x : np.array
        lasso solution
    D : np.array
        dictionary

    Returns
    -------
    np.array
        Jacobian
    """
    jacobian = np.zeros((x.shape[0], D.shape[0], D.shape[1]))
    support = (np.abs(x) > 0)
    if np.sum(np.abs(support)) > 0:
        inv = -np.linalg.inv((D[:, support].T @ D[:, support]))
        for i in range(D.shape[0]):
            a = D[i].reshape(-1, 1) @ x.reshape(1, -1)
            b = (D[i].T @ x - y[i]) * np.eye(x.shape[0])
            M = (a + b)
            jacobian[:, i, :][support] = (inv @ M[support, :])
    return jacobian


def distance_support(x1, x2):
    """
    Hamming distance between the supports

    Parameters
    ----------
    x1 : np.array
        sample 1
    x2 : np.array
        sample 2

    Returns
    -------
    int
        Hamming distance between x1 and x2
    """
    s1 = (np.abs(x1[:, 0]) > 0).astype(int)
    s2 = (np.abs(x2[:, 0]) > 0).astype(int)
    return np.abs(s1 - s2).sum()


def process_row(row, y, x, path, tj, lipschitz, d_init, n_iter):
    m = path[0].shape[0]
    step = 1 / lipschitz

    # Computation of one row of the Jacobian
    jacobian = np.zeros((m, d_init.shape[1]))
    error = [np.linalg.norm(tj[:, row, :] - jacobian)]
    distance = [distance_support(path[0], x)]
    for i in range(1, n_iter):
        jacobian -= step * (d_init[row].reshape(-1, 1)
                            @ path[i-1].reshape(1, -1)
                            + (d_init[row] @ path[i-1] - y[row]) * np.eye(m)
                            + d_init.T @ d_init @ jacobian)
        support = np.zeros(m)
        support[(np.abs(path[i]) > 0).reshape(1, -1)[0]] = 1
        jacobian *= support.reshape(-1, 1)
        error.append(np.linalg.norm(tj[:, row, :] - jacobian))
        distance.append(distance_support(path[i], x))
    return np.array(error), np.array(distance)


def process_row_backprop(row, path, lipschitz, d_init, y, K, iterations, tj):
    error = []
    for i in iterations:
        nth_jac = compute_nth_jac(i, path, row, lipschitz, d_init, y, K)
        error.append(np.linalg.norm(tj[:, row, :] - nth_jac))
    return np.array(error)


def process_dico(data, D, n_iter, n_examples, lambd, examples):
    # Evaluation of the error
    lipschitz = compute_lipschitz(D)
    dim = data.shape[0]

    error_avg = np.zeros((n_examples, n_iter))
    distance_avg = np.zeros((n_examples, n_iter))

    for ex in range(len(examples)):
        errors = np.zeros((dim, n_iter))
        distances = np.zeros((dim, n_iter))

        # ISTA
        y = data[:, examples[ex]].reshape(-1, 1)
        x_result_fista, path = ista(y, D, n_iter, lipschitz, lambd=lambd)

        # True Jacobian
        tj = true_jac(y, x_result_fista[:, 0], D)
        if np.linalg.norm(tj) < 1e5:

            # Computation of one row of the Jacobian
            with parallel_backend('loky', inner_max_num_threads=1):
                results = Parallel(n_jobs=N_JOBS)(
                    delayed(process_row)(row, y, x_result_fista, path,
                                         tj, lipschitz, D, n_iter)
                    for row in range(dim)
                    )

            for i in range(len(results)):
                errors[i] = results[i][0]
                distances[i] = results[i][1]

            error_avg[ex] = errors.mean(axis=0)
            distance_avg[ex] = distances.mean(axis=0)

    return error_avg, distance_avg


def multiple_backprop(data, D, iterations, iterations_backprop,
                      n_examples, lambd, examples):

    lipschitz = compute_lipschitz(D)
    dim = data.shape[0]

    error_avg_K = np.zeros(
        (n_examples, len(iterations_backprop), len(iterations))
        )

    for ex in range(len(examples)):

        # ISTA
        y = data[:, examples[ex]].reshape(-1, 1)
        x_result_fista, path = ista(y, D, n_iter, lipschitz, lambd=lambd)

        # True Jacobian
        tj = true_jac(y, x_result_fista[:, 0], D)
        if np.linalg.norm(tj) < 1e5:
            errors = np.zeros(
                (len(iterations_backprop), tj.shape[1], iterations.shape[0])
                )
            for i in range(len(iterations_backprop)):
                K = iterations_backprop[i]
                with parallel_backend('loky', inner_max_num_threads=1):
                    results = Parallel(n_jobs=N_JOBS)(
                        delayed(process_row_backprop)(row, path, lipschitz, D,
                                                      y, K, iterations, tj)
                        for row in range(dim)
                        )
                for j in range(len(results)):
                    errors[i, j] = results[j]
            error_avg_K[ex] = errors.mean(axis=1)

    return error_avg_K


def generate_data(dico, N, k=0.3):
    """
    Generate data from dictionary

    Parameters
    ----------
    dico : np.array
        dictionary
    N : int
        number of samples
    k : float, optional
        sparsity, by default 0.3

    Returns
    -------
    (np.array, np.array)
        signal, sparse codes
    """
    d = dico.shape[1]
    X = (RNG.random((d, N)) > (1-k)).astype(float)
    X *= RNG.normal(scale=1, size=(d, N))
    return dico @ X, X


# Data
d = 50
P = 30
N = 100
sigma = 0.1

A = RNG.normal(size=(P, d)) / P
A /= np.sqrt(np.sum(A**2, axis=0))

data_x, codes = generate_data(A, N)

noise = RNG.normal(scale=sigma, size=data_x.shape)
data = data_x + noise


# Optimization parameters
lambd = 0.1
sigma_dic = A.std() * 0.5
current_dictionary = A + RNG.normal(scale=sigma_dic, size=A.shape)
examples = np.arange(N_EXAMPLES)
n_iter = int(1e4) + 1


# Jacobian errors
print("Jacobian errors")
errors = []
distances = []

errors, distances = process_dico(data, current_dictionary,
                                 n_iter, N_EXAMPLES, lambd,
                                 examples)
iterations = np.arange(n_iter)

np.save(str(RESULTS / 'jac_iterations.npy'), iterations)
np.save(str(RESULTS / 'jac_error.npy'), errors)
np.save(str(RESULTS / 'jac_distance_support.npy'), distances)


# Multiple backprop depth
print("Multiple backprop depth")
iterations_backprop = [200, 50, 20]
iterations = np.unique(np.logspace(0, np.log10(n_iter - 1), dtype=int, num=30))

errors_K = []
errors_K = multiple_backprop(data, current_dictionary, iterations,
                             iterations_backprop, N_EXAMPLES, lambd,
                             examples)

np.save(str(RESULTS / 'jac_iterations_backprop.npy'), iterations_backprop)
np.save(str(RESULTS / 'jac_iterations_2.npy'), iterations)
np.save(str(RESULTS / 'jac_error_backprop.npy'), errors_K)
