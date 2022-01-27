import numpy as np
import pickle
import torch

from tqdm import tqdm
from pathlib import Path
from plipy.ddl import DeepDictionaryLearning, AMDictionaryLearning
from plipy.ddl_sto import StoDeepDictionaryLearning


RESULTS = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(2022)
N_EXAMPLES = 10
DEVICE = "cuda:0"


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

# Optim parameters
iteration = 30
lambd = 0.1
time_limit = 100

n_samples = [100, 500, 2000, 10000]
all_results = {}

for j in tqdm(range(N_EXAMPLES)):

    A = RNG.normal(size=(P, d)) / P
    A /= np.sqrt(np.sum(A**2, axis=0))

    data_x, codes = generate_data(A, N)

    noise = RNG.normal(scale=sigma, size=data_x.shape)
    data = data_x + noise

    current_dictionary = RNG.normal(size=(P, d)) / P
    current_dictionary /= np.sqrt(np.sum(current_dictionary**2, axis=0))

    results = {"AM": {}, "DDL": {}, "DDL_sto": {}, "dico": A}

    amdl = AMDictionaryLearning(d, 1000, lambd,
                                init_D=current_dictionary, device=DEVICE)
    amdl.keep_dico = True
    amdl.fit(data)
    results["AM"] = {}
    results["AM"]["losses"] = amdl.path_loss
    results["AM"]["times"] = amdl.path_times
    results["AM"]["path"] = amdl.path_optim
    results["AM"]["dico"] = amdl.get_prior()

    torch.cuda.reset_peak_memory_stats()
    ddl = DeepDictionaryLearning(d, iteration, lambd, learn_steps=False,
                                 init_D=current_dictionary, device=DEVICE)
    ddl.keep_dico = True
    ddl.fit(data)
    results["DDL"] = {}
    results["DDL"]["losses"] = ddl.path_loss
    results["DDL"]["times"] = ddl.path_times
    results["DDL"]["path"] = ddl.path_optim
    results["DDL"]["dico"] = ddl.get_prior()
    results["DDL"]["memory"] = torch.cuda.max_memory_allocated()

    for i in n_samples:
        torch.cuda.reset_peak_memory_stats()
        ddl_sto = StoDeepDictionaryLearning(
            d, iteration, lambd, learn_steps=False,
            init_D=current_dictionary, device=DEVICE
        )
        ddl_sto.keep_dico = True
        ddl_sto.fit(
            data, mini_batch_size=int(i), epochs=100,
            iterations_per_epoch=100, time_limit=time_limit
        )
        results["DDL_sto"][i] = {}
        results["DDL_sto"][i]["losses"] = ddl_sto.path_loss
        results["DDL_sto"][i]["times"] = ddl_sto.path_times
        results["DDL_sto"][i]["path"] = ddl_sto.path_optim
        results["DDL_sto"][i]["dico"] = ddl_sto.get_prior()
        results["DDL_sto"][i]["memory"] = torch.cuda.max_memory_allocated()

    all_results[j] = results

np.save(str(RESULTS / "optim_sto_n_samples.npy"), n_samples)

with open(str(RESULTS / 'optim_sto_results.pickle'), 'wb') as file1:
    pickle.dump(all_results, file1)
