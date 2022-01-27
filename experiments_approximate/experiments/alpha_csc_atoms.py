import os
import mne
import numpy as np
import time

from alphacsc import GreedyCDL
from alphacsc.utils import split_signal


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

# Regularization parameter which control sparsity
reg = 0.1

cdl = GreedyCDL(
    # Shape of the dictionary
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    rank1=True,
    uv_constraint='separate',
    # apply a temporal window reparametrization
    window=True,
    # at the end, refit the activations with fixed support and no reg to unbias
    unbiased_z_hat=True,
    # Initialize the dictionary with random chunk from the data
    D_init='chunk',
    # rescale the regularization parameter to be a percentage of lambda_max
    lmbd_max="scaled",
    reg=reg,
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=100,
    eps=1e-4,
    # solver for the z-step
    solver_z="lgcd",
    solver_z_kwargs={'tol': 1e-3,
                     'max_iter': 100000},
    # solver for the d-step
    solver_d='alternate_adaptive',
    solver_d_kwargs={'max_iter': 300},
    # sort atoms by explained variances
    sort_atoms=True,
    # Technical parameters
    verbose=1,
    random_state=0,
    n_jobs=n_jobs)


X_split = split_signal(X, n_splits=n_splits, apply_window=True)

start = time.time()
cdl.fit(X_split)
print("Time alphacsc: ", time.time() - start)

u_cdl = cdl.u_hat_
v_cdl = cdl.v_hat_

np.save("u_cdl.npy", u_cdl)
np.save("v_cdl.npy", v_cdl)
