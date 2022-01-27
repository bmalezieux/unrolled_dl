import numpy as np
import pickle

from tqdm import tqdm
from pathlib import Path
from plipy.ddl import DeepDictionaryLearning, AMDictionaryLearning


RESULTS = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(2022)
N_SAMPLES = 50


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
d = 50
P = 30
N = 1000
sigma = 0.1

# Optim parameters
iterations = np.unique(np.logspace(np.log10(5), 3, num=50, dtype=int))
lambd = 0.1

results = {}

for j in tqdm(range(N_SAMPLES)):

    A = RNG.normal(size=(P, d)) / P
    A /= np.sqrt(np.sum(A**2, axis=0))

    data_x, codes = generate_data(A, N)

    noise = RNG.normal(scale=sigma, size=data_x.shape)
    data = data_x + noise

    sigma_dic = A.std() * 0.5
    current_dictionary = A + RNG.normal(scale=sigma_dic, size=A.shape)

    am_dico = {"paths": {}, "losses": {}, "times": {}}
    ddl_dico = {"paths": {}, "losses": {}, "times": {}}
    ddl_steps_dico = {"paths": {}, "losses": {}, "times": {}}

    for i in iterations:
        am = AMDictionaryLearning(d, i, lambd, init_D=current_dictionary)
        am.keep_dico = True
        am.fit(data)
        am_dico["paths"][i] = am.path_optim
        am_dico["losses"][i] = am.path_loss
        am_dico["times"][i] = am.path_times

        ddl = DeepDictionaryLearning(d, i, lambd, learn_steps=False,
                                     init_D=current_dictionary)
        ddl.keep_dico = True
        ddl.fit(data)
        ddl_dico["paths"][i] = ddl.path_optim
        ddl_dico["losses"][i] = ddl.path_loss
        ddl_dico["times"][i] = ddl.path_times

        if i <= 100:

            ddl_steps = DeepDictionaryLearning(d, i, lambd, learn_steps=True,
                                               init_D=current_dictionary)
            ddl_steps.keep_dico = True
            ddl_steps.fit(data)
            ddl_steps_dico["paths"][i] = ddl_steps.path_optim
            ddl_steps_dico["losses"][i] = ddl_steps.path_loss
            ddl_steps_dico["times"][i] = ddl_steps.path_times

        else:
            ddl_steps_dico["paths"][i] = []
            ddl_steps_dico["losses"][i] = []
            ddl_steps_dico["times"][i] = []

    results[j] = {
        "am": am_dico,
        "ddl": ddl_dico,
        "ddl_steps": ddl_steps_dico,
        "dico": A
        }

np.save(str(RESULTS / "optim_iterations.npy"), iterations)

with open(str(RESULTS / 'optim_paths.pickle'), 'wb') as file1:
    pickle.dump(results, file1)
