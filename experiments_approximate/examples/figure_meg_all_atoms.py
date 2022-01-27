import os
import mne
import numpy as np
import matplotlib.pyplot as plt


# sampling frequency. The signal will be resampled to match this.
sfreq = 150.

# Define the shape of the dictionary
n_atoms = 40
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

# Regularization parameter which controls sparsity
reg = 0.5

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

D = np.load("dico_1rank.npy")
u = np.load("weights_channels.npy")


plotted_atoms = range(n_atoms)

# number of plots by atom
n_plots = 2
n_columns = min(6, len(plotted_atoms))
split = int(np.ceil(len(plotted_atoms) / n_columns))
figsize = (4 * n_columns, 3 * n_plots * split)
fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)

for ii, kk in enumerate(plotted_atoms):

    # Select the axes to display the current atom
    i_row, i_col = ii // n_columns, ii % n_columns
    it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

    # Select the current atom
    v_k = D[kk, 0, :]
    u_k = u[kk, :, 0]

    # Plot the spatial map of the atom using mne topomap
    ax = next(it_axes)
    ax.set_title('Atom % d' % kk, pad=0)

    mne.viz.plot_topomap(data=u_k, pos=info, axes=ax, show=False)
    if i_col == 0:
        ax.set_ylabel('Spatial', labelpad=28)

    # Plot the temporal pattern of the atom
    ax = next(it_axes)
    ax.plot(v_k)
    if i_col == 0:
        ax.set_ylabel('Temporal')

# save figure
fig.tight_layout()
path_fig = 'sample_all_atoms.pdf'
plt.savefig(path_fig, dpi=300, bbox_inches='tight')
plt.close()
