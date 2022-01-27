import os
import mne
import numpy as np
import time
import pandas as pd
import itertools

from plipy.ddl_sto import StoDeepCDL1Rank
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate
from joblib import Memory
from tqdm import tqdm

mem = Memory(location='.', verbose=0)
N_EXAMPLES = 10


def cost_matrix_v(D, Dref):
    C = np.zeros((D.shape[0], Dref.shape[0]))
    for i in range(D.shape[0]):
        for j in range(Dref.shape[0]):
            C[i, j] = correlate(D[i, 0], Dref[j, 0]).max()
    return C


def recovery_score(D, Dref, u=True):
    """
    Comparison between a learnt prior and the truth
    """
    try:
        if u:
            cost_matrix = np.abs(Dref.T @ D)
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            score = cost_matrix[row_ind, col_ind].sum() / np.min([D.shape[1], Dref.shape[1]])
        else:
            cost_matrix = cost_matrix_v(D, Dref)
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            score = cost_matrix[row_ind, col_ind].sum()
    except:
        score = 0
    return score



@mem.cache
def run_test(params, num_exp):
    lambd = params["lambd"]
    mbs = params["mbs"]
    window = params["window"]
    epoch_steps = params["epochs_steps"]
    epoch = params["epochs"]
    iter_per_epoch = params["iter_per_epoch"]

    reco_u = np.zeros(num_exp)
    reco_v = np.zeros(num_exp)
    times = np.zeros(num_exp)

    for i in range(num_exp):

        start = time.time()
        dcdl = StoDeepCDL1Rank(n_components=n_atoms, n_iter=epoch, lambd=lambd,
                               kernel_size=n_times_atom, device="cuda:3")
        dcdl.fit(X, window=window, mini_batch_size=mbs,
                 iterations_per_epoch=iter_per_epoch, c=0.001,
                 epochs=epoch, epochs_step_size=epoch_steps)

        times[i] = time.time() - start
        u_ddl, v_ddl = dcdl.get_prior()

        v_ddl /= np.linalg.norm(v_ddl, axis=2, keepdims=True)

        reco_u[i] = recovery_score(u_ddl[:, :, 0].T, u_cdl.T, u=True)
        reco_v[i] = recovery_score(v_ddl[:, 0, :].T, v_cdl.T, u=False)

    results = {
        "time_avg": times.mean(),
        "recovery_u_avg": reco_u.mean(),
        "recovery_v_avg": reco_v.mean(),
        "time_std": times.std(),
        "recovery_u_std": reco_u.std(),
        "recovery_v_std": reco_v.std()
    }

    return results


# sampling frequency. The signal will be resampled to match this.
sfreq = 150.

# Define the shape of the dictionary
n_atoms = 40
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

# number of processors for parallel computing
n_jobs = 10

# To accelerate the run time of this example, we split the signal in n_slits.
# The number of splits should actually be the smallest possible to avoid
# introducing border artifacts in the learned atoms and it should be not much
# larger than n_jobs.
n_splits = 10

print("Loading the data...", end='', flush=True)
data_path = mne.datasets.sample.data_path()
subjects_dir = os.path.join(data_path, "subjects")
data_dir = os.path.join(data_path, 'MEG', 'sample')
file_name = os.path.join(data_dir, 'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(file_name, preload=True, verbose=False)
raw.pick_types(meg='grad', eeg=False, eog=False, stim=True)
print('done')

print("Preprocessing the data...", end='', flush=True)
raw.notch_filter(np.arange(60, 181, 60), n_jobs=n_jobs, verbose=False)
raw.filter(2, None, n_jobs=n_jobs, verbose=False)
raw = raw.resample(sfreq, npad='auto', n_jobs=n_jobs, verbose=False)
print('done')

X = raw.get_data(picks=['meg'])
info = raw.copy().pick_types(meg=True).info  # info of the loaded channels

print(info)
np.save("data_meg.npy", X)

u_cdl = np.load("u_cdl_modified.npy")
v_cdl = np.load("v_cdl_modified.npy")

hyperparams = {
    "lambd": [0.1, 0.3, 0.5],
    "mbs": [5, 10, 20, 40],
    "window": [1000, 2000],
    "epochs_steps": [0, 5],
    "epochs": [10],
    "n_iter": [30],
    "iter_per_epoch": [10]
}

keys, values = zip(*hyperparams.items())
permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
results = {}

for params in tqdm(permuts_params):
    results_exp = run_test(params, N_EXAMPLES)

    for key in params:
        if key in results:
            results[key].append(params[key])
        else:
            results[key] = [params[key]]

    for key in results_exp:
        if key in results:
            results[key].append(results_exp[key])
        else:
            results[key] = [results_exp[key]]

results = pd.DataFrame(results)
results.to_csv("results_denoising.csv")
