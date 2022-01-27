import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

from scipy.optimize import linear_sum_assignment
from pathlib import Path


FIGURES = Path(__file__).resolve().parents[1] / "figures"
RESULTS = Path(__file__).resolve().parents[1] / "results"


n_samples = np.load(str(RESULTS / 'optim_sto_n_samples.npy'))

with open(str(RESULTS / 'optim_sto_results.pickle'), 'rb') as file1:
    results = pickle.load(file1)


def recovery_score(D, Dref):
    """
    Comparison between a learnt prior and the truth
    """
    try:
        cost_matrix = np.abs(Dref.T@D)

        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        score = cost_matrix[row_ind, col_ind].sum() / D.shape[1]
    except:
        score = 0

    return score


plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("text", usetex=False)
plt.rc('font', family='serif', size=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

plt.subplots(figsize=(4, 2))

n_exp = len(results.keys())
cmap = matplotlib.cm.get_cmap('viridis')
samples_array = np.log10(np.array(list(n_samples) + [20000]))
colors = [cmap(i) for i in (samples_array - samples_array.min()) / (samples_array.max() - samples_array.min()) ]

# DDL Sto
cpt = 0
memories = []

for j in n_samples:
    hist_data = []
    for i in range(n_exp):
        recoveries = []
        times = results[i]["DDL_sto"][j]["times"]
        for elt in results[i]["DDL_sto"][j]["path"]:
            recoveries.append(recovery_score(elt, results[i]["dico"]))
        if np.array(recoveries).max() > 0.95:
            min_t = np.array(times)[np.where(np.array(recoveries) > 0.95)].min()
            hist_data.append(min_t)
        else:
            hist_data.append(50)

    plt.bar(np.arange(1) - 0.2 + 0.1 * cpt, np.mean(hist_data), color=colors[cpt], label=str(j), width=0.1, zorder=-1)
    plt.scatter((np.arange(1) - 0.2 + 0.1 * cpt) * np.ones(len(hist_data)), hist_data, color="black", zorder=1, s=10, marker='_')
    cpt += 1

# DDL

hist_data = []
for i in range(n_exp):
    recoveries = []
    times = results[i]["DDL"]["times"]
    for elt in results[i]["DDL"]["path"]:
        recoveries.append(recovery_score(elt, results[i]["dico"]))
    if np.array(recoveries).max() > 0.95:
        min_t = np.array(times)[np.where(np.array(recoveries) > 0.95)].min()
        hist_data.append(min_t)

plt.bar(np.arange(1) - 0.2 + 0.1 * cpt, np.mean(hist_data), color="red", label="DDL", width=0.1, zorder=-1)
plt.scatter((np.arange(1) - 0.2 + 0.1 * cpt) * np.ones(len(hist_data)), hist_data, color="black", zorder=1, s=10, marker='_')

cpt += 1
plt.bar(np.arange(1) - 0.2 + 0.1 * cpt, [50], color=(0.8, 0.8, 0.8), label="Oracle DL", width=0.1)
plt.text(0.28, 18, "Timeout", rotation=90)

plt.title("Time to reach a score of 0.95", fontsize=10)
plt.grid(axis="y")
plt.ylabel("Time (s)")
plt.ylim([0, 50])
plt.xticks(list(np.arange(1) - 0.2 + 0.1 * np.arange(cpt+1)), [str(i) for i in n_samples] + ["DDL", "Oracle DL"], fontsize=8)
plt.xlabel("Minibatch size")
plt.tight_layout()
plt.savefig(str(FIGURES / "optim_sto_hist.pdf"))
plt.clf()
