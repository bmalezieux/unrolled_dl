import os
import mne
import numpy as np
import time

from plipy.ddl_sto import StoDeepCDL1Rank

# sampling frequency. The signal will be resampled to match this.
sfreq = 150.

# Define the shape of the dictionary
n_atoms = 40
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

n_jobs = 5

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

##### Rank 1 all channels #####

start = time.time()
dcdl = StoDeepCDL1Rank(n_components=n_atoms, n_iter=30, lambd=0.3,
                       kernel_size=n_times_atom, device=None)
dcdl.fit(X, window=1000, mini_batch_size=20, iterations_per_epoch=10,
         c=0.001, epochs=10, epochs_step_size=0)

print("Time: ", time.time() - start)
u, D = dcdl.get_prior()
np.save("dico_1rank.npy", D)
np.save("weights_channels.npy", u)
