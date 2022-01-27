import numpy as np
import os
import mne
import time

from ksvd_cdl import KSVD_CDL


# sampling frequency. The signal will be resampled to match this.
sfreq = 150.

# Define the shape of the dictionary
n_atoms = 40
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

# number of processors for parallel computing
n_jobs = 10

print("Loading the data...", end='', flush=True)
data_path = mne.datasets.sample.data_path("/storage/store2/work/bmalezie/")
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
print(X.shape)

start = time.time()
ksvd = KSVD_CDL(n_atoms=n_atoms, n_times_atom=n_times_atom,
                max_iter=100000,
                n_iter=40, correlation='fast')
ksvd.fit(X)
print(f"Time: {time.time() - start}")

np.save("u_ksvd.npy", ksvd.u_hat_)
np.save("v_ksvd.npy", ksvd.v_hat_)

import matplotlib.pyplot as plt

# preselected atoms of interest
plotted_atoms = range(n_atoms)

n_plots = 3  # number of plots by atom
n_columns = min(6, len(plotted_atoms))
split = int(np.ceil(len(plotted_atoms) / n_columns))
figsize = (4 * n_columns, 3 * n_plots * split)
fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)
for ii, kk in enumerate(plotted_atoms):

    # Select the axes to display the current atom
    print("\rDisplaying {}-th atom".format(kk), end='', flush=True)
    i_row, i_col = ii // n_columns, ii % n_columns
    it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

    # Select the current atom
    u_k = ksvd.u_hat_[kk]
    v_k = ksvd.v_hat_[kk]

    # Plot the spatial map of the atom using mne topomap
    ax = next(it_axes)
    mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
    ax.set(title="Spatial pattern %d" % (kk, ))

    # Plot the temporal pattern of the atom
    ax = next(it_axes)
    t = np.arange(n_times_atom) / sfreq
    ax.plot(t, v_k)
    ax.set_xlim(0, n_times_atom / sfreq)
    ax.set(xlabel='Time (sec)', title="Temporal pattern %d" % kk)

    # Plot the power spectral density (PSD)
    ax = next(it_axes)
    psd = np.abs(np.fft.rfft(v_k, n=256)) ** 2
    frequencies = np.linspace(0, sfreq / 2.0, len(psd))
    ax.semilogy(frequencies, psd, label='PSD', color='k')
    ax.set(xlabel='Frequencies (Hz)', title="Power spectral density %d" % kk)
    ax.grid(True)
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-4, 1e2)
    ax.legend()
print("\rDisplayed {} atoms".format(len(plotted_atoms)).ljust(60))

fig.tight_layout()
fig.savefig('atoms_ksvd.pdf')

ksvd.pobj_.to_pickle('pobj.pkl')
plt.figure()
plt.plot(ksvd.pobj_['time'].cumsum() / 3600, ksvd.pobj_['loss'])
plt.xlabel('time [hours]')
plt.ylabel('Loss')
plt.savefig('objective_evolution.pdf')
plt.show()
