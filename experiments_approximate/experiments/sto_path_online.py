import numpy as np
import time
import pickle

from scipy.optimize import linear_sum_assignment
from spams import trainDL
from plipy.ddl_sto import StoDeepDictionaryLearning
from tqdm import tqdm
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(2022)
N_EXAMPLES = 10
DEVICE = "cuda:0"


def recovery_score(D, Dref):
    """
    Comparison between a learnt prior and the truth
    """
    try:
        cost_matrix = np.abs(Dref.T@D)

        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        score = cost_matrix[row_ind, col_ind].sum() / np.min([D.shape[1], Dref.shape[1]])
    except:
        score = 0

    return score


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


# Data parameters
d = 100
P = 50
N = int(1e5)
sigma = 0.1

lambdas = np.arange(0.1, 1.1, 0.1)
mini_batch_size = 2000
time_limit = 50
iteration = 30

# DDL

all_results_ddl = {}

for j in tqdm(range(N_EXAMPLES)):

    results = {}

    for lambd in lambdas:

        A = RNG.normal(size=(P, d)) / P
        A /= np.sqrt(np.sum(A**2, axis=0))

        data_x, codes = generate_data(A, N)

        noise = RNG.normal(scale=sigma, size=data_x.shape)
        data = data_x + noise

        ddl_sto = StoDeepDictionaryLearning(d, iteration, lambd,
                                            learn_steps=False,
                                            device=DEVICE)
        ddl_sto.keep_dico = True
        ddl_sto.fit(
            data, mini_batch_size=mini_batch_size, epochs=100,
            iterations_per_epoch=100, time_limit=time_limit
        )

        for i in range(len(ddl_sto.path_optim)):
            if recovery_score(ddl_sto.path_optim[i], A) > 0.95:
                results[lambd] = ddl_sto.path_times[i]
                break

    all_results_ddl[j] = results

with open(str(RESULTS / 'optim_online_ddl_results.pickle'), 'wb') as file1:
    pickle.dump(all_results_ddl, file1)


# Online DL

all_results_online = {}

for j in tqdm(range(N_EXAMPLES)):

    results = {}
    for lambd in lambdas:

        A = RNG.normal(size=(P, d)) / P
        A /= np.sqrt(np.sum(A**2, axis=0))

        data_x, codes = generate_data(A, N)

        noise = RNG.normal(scale=sigma, size=data_x.shape)
        data = data_x + noise

        for iteration in np.arange(100, 3000, 100, dtype=int):

            param = {
                'K': d,
                'verbose': False,
                'lambda1': lambd,
                'numThreads': 10,
                "batchsize": mini_batch_size,
                'iter': int(iteration),
                'mode': 2
            }

            start = time.time()
            dico = np.array(trainDL(np.asfortranarray(data), **param))
            stop = time.time()

            if recovery_score(dico, A) > 0.95:
                results[lambd] = stop - start
                break

    all_results_online[j] = results


with open(str(RESULTS / 'optim_online_am_results.pickle'), 'wb') as file1:
    pickle.dump(all_results_online, file1)


np.save(str(RESULTS / "optim_online_lambdas.npy"), lambdas)
