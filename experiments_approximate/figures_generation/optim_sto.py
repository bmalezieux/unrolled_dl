import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

from scipy.optimize import linear_sum_assignment
from pathlib import Path
from pygam import LinearGAM


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
plt.rc('font', family='serif', size=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

plt.figure(figsize=(4.5, 3.5))

n_exp = len(results.keys())
cmap = matplotlib.cm.get_cmap('viridis')
samples_array = np.log10(np.array(list(n_samples) + [20000]))
colors = [cmap(i) for i in (samples_array - samples_array.min()) / (samples_array.max() - samples_array.min())]
# colors = ["purple", "darkblue", "darkgreen", "darkgoldenrod", "brown", "grey"]


# DDL Sto

for j in n_samples:
    recoveries = []
    times = []
    for i in range(n_exp):
        times += results[i]["DDL_sto"][j]["times"]
        for elt in results[i]["DDL_sto"][j]["path"]:
            recoveries.append(recovery_score(elt, results[i]["dico"]))
    X = np.array(times).reshape(-1, 1)
    y = np.array(recoveries)
    gam = LinearGAM(n_splines=25).gridsearch(X, y)
    XX = gam.generate_X_grid(term=0, n=500)

    plt.plot(XX,
             gam.predict(XX),
             color=colors[np.where(n_samples == j)[0][0]],
             linewidth=0.5)
    conf = gam.prediction_intervals(XX, width=0.95)
    plt.fill_between(XX[:, 0],
                     conf[:, 0],
                     conf[:, 1],
                     color=colors[np.where(n_samples == j)[0][0]],
                     alpha=0.2)

# DDL

recoveries = []
times = []
for i in range(n_exp):
    times += results[i]["DDL"]["times"]
    for elt in results[i]["DDL"]["path"]:
        recoveries.append(recovery_score(elt, results[i]["dico"]))

X = np.array(times).reshape(-1, 1)
y = np.array(recoveries)
gam = LinearGAM(n_splines=25).gridsearch(X, y)
XX = gam.generate_X_grid(term=0, n=500)

plt.plot(XX, gam.predict(XX), color="red", linewidth=1)
conf = gam.prediction_intervals(XX, width=0.95)
plt.fill_between(XX[:, 0], conf[:, 0], conf[:, 1], color="red", alpha=0.2)


# Full AM

# recoveries = []
# times = []
# for i in range(n_exp):
#     times += results[i]["AM"]["times"]
#     for elt in results[i]["AM"]["path"]:
#         recoveries.append(recovery_score(elt, results[i]["dico"]))

# X = np.array(times).reshape(-1, 1)
# y = np.array(recoveries)
# gam = LinearGAM(n_splines=25).gridsearch(X, y)
# XX = gam.generate_X_grid(term=0, n=500)

recoveries = []
for i in range(n_exp):
    recoveries.append(recovery_score(results[i]["AM"]["dico"], results[i]["dico"]))

# plt.plot(XX, gam.predict(XX), color="black", linewidth=1)
# conf = gam.prediction_intervals(XX, width=0.95)
# plt.fill_between(XX[:, 0], conf[:, 0], conf[:, 1], color="black", alpha=0.2)

x = np.array(recoveries).mean()

plt.plot([0, 100], [x, x], '--', color="black", linewidth=1)

plt.legend(labels=[str(elt) for elt in n_samples] + ["Full batch", "Complete AM"],
           labelcolor=colors[:-1] + ['red', 'black'], title="Minibatch size",
           loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1), fontsize=12, title_fontsize=12)
plt.ylabel("Rec. score")
plt.ylim([0.4, 1])
plt.xlabel("Time (s)")
plt.xlim([0, 50])
plt.tight_layout()
plt.savefig(str(FIGURES / "optim_sto_score.pdf"))
plt.clf()
