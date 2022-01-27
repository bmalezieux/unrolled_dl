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

plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc('font', family='serif', size=22)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

# preselected atoms of interest
plotted_atoms = [1, 22, 5]

n_plots = 2  # number of plots by atom
n_columns = min(6, len(plotted_atoms))
split = int(np.ceil(len(plotted_atoms) / n_columns))
figsize = (4 * n_columns, 3 * n_plots * split)
fig, axes = plt.subplots(n_plots * split, n_columns, figsize=figsize)
for ii, kk in enumerate(plotted_atoms):

    # Select the axes to display the current atom
    print("\rDisplaying {}-th atom".format(ii), end='', flush=True)
    i_row, i_col = ii // n_columns, ii % n_columns
    it_axes = iter(axes[i_row * n_plots:(i_row + 1) * n_plots, i_col])

    # Select the current atom
    v_k = D[kk, 0, :]
    u_k = u[kk, :, 0]

    # Plot the spatial map of the atom using mne topomap
    ax = next(it_axes)
    mne.viz.plot_topomap(u_k, info, axes=ax, show=False)
    ax.set_title("Spatial pattern %d" % (ii, ), fontsize=22)

    # Plot the temporal pattern of the atom
    ax = next(it_axes)
    t = np.arange(n_times_atom) / sfreq
    ax.plot(t, v_k)
    ax.set_xlim(0, n_times_atom / sfreq)
    ax.set_xlabel("Time (s)")
    ax.set_title("Temporal pattern %d" % ii, fontsize=22)

    # # Plot the power spectral density (PSD)
    # ax = next(it_axes)
    # psd = np.abs(np.fft.rfft(v_k, n=256)) ** 2
    # frequencies = np.linspace(0, sfreq / 2.0, len(psd))
    # ax.semilogy(frequencies, psd, label='PSD', color='k')
    # ax.set(xlabel='Frequencies (Hz)', title="Power spectral density %d" % kk)
    # ax.grid(True)
    # ax.set_xlim(0, 30)
    # ax.set_ylim(1e-4, 1e2)
    # ax.legend()
print("\rDisplayed {} atoms".format(len(plotted_atoms)).rjust(40))

fig.tight_layout()
plt.savefig("meg_figure.pdf")
