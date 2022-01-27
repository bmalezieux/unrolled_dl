import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from scipy.optimize import linear_sum_assignment
from pathlib import Path

try:
    os.mkdir("../figures")
except OSError:
    pass


FIGURES = Path(__file__).resolve().parents[1] / "figures"
RESULTS = Path(__file__).resolve().parents[1] / "results"


def recovery_score(D, Dref):
    """
    Comparison between a learnt prior and the truth
    """
    cost_matrix = np.abs(Dref.T@D)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    score = cost_matrix[row_ind, col_ind].sum() / D.shape[1]

    return score


iterations = np.load(str(RESULTS / 'optim_iterations.npy'))
with open(str(RESULTS / 'optim_paths.pickle'), 'rb') as file1:
    results = pickle.load(file1)

len_ddl_steps = np.argwhere(iterations < 100)[-1][0]

plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("text", usetex=False)
plt.rc('font', family='serif', size=11)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)


fig, axs = plt.subplots(1, 3, figsize=(6.875, 1.875))
lines = []
lines_label = []


# Gradient steps
ddl_steps = np.zeros((len(results.keys()), len(iterations)))
am_steps = np.zeros((len(results.keys()), len(iterations)))
ddl_steps_steps = np.zeros((len(results.keys()), len(iterations)))

for j in results.keys():
    for i in range(len(iterations)):
        am_steps[j, i] = len(results[j]["am"]["paths"][iterations[i]])
        ddl_steps[j, i] = len(results[j]["ddl"]["paths"][iterations[i]])
        ddl_steps_steps[j, i] = len(results[j]["ddl_steps"]["paths"][iterations[i]])

lines.append(axs[0].plot(iterations, am_steps.mean(axis=0))[0])
lines_label.append("AM")
axs[0].fill_between(iterations,
                    np.quantile(am_steps, 0.1, axis=0),
                    np.quantile(am_steps, 0.9, axis=0),
                    alpha=0.2)

lines.append(axs[0].plot(iterations, ddl_steps.mean(axis=0))[0])
lines_label.append("DDL")
axs[0].fill_between(iterations,
                    np.quantile(ddl_steps, 0.1, axis=0),
                    np.quantile(ddl_steps, 0.9, axis=0),
                    alpha=0.2)

lines.append(axs[0].plot(iterations[:len_ddl_steps+1], ddl_steps_steps.mean(axis=0)[:len_ddl_steps+1])[0])
lines_label.append("DDL + steps")

axs[0].fill_between(iterations[:len_ddl_steps+1],
                    np.quantile(ddl_steps_steps[:, :len_ddl_steps+1], 0.1, axis=0),
                    np.quantile(ddl_steps_steps[:, :len_ddl_steps+1], 0.9, axis=0),
                    alpha=0.2)

axs[0].set_xscale("log")
axs[0].set_ylabel("Number")
axs[0].set_xlabel("Iterations N")
axs[0].set_title("Gradient steps", fontsize=11)

fig.legend(lines, lines_label, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.2))


# Loss
ddl_loss = np.zeros((len(results.keys()), len(iterations)))
am_loss = np.zeros((len(results.keys()), len(iterations)))
ddl_steps_loss = np.zeros((len(results.keys()), len(iterations)))

for j in results.keys():
    for i in range(len(iterations)):
        am_loss[j, i] = results[j]["am"]["losses"][iterations[i]][-1]
        ddl_loss[j, i] = results[j]["ddl"]["losses"][iterations[i]][-1]
        try:
            ddl_steps_loss[j, i] = results[j]["ddl_steps"]["losses"][iterations[i]][-1]
        except:
            ddl_steps_loss[j, i] = 0

best_loss = (np.min(np.concatenate([am_loss.min(axis=1)[:, None], ddl_loss.min(axis=1)[:, None]], axis=1), axis=1) - 1e-3)[:, None]

axs[1].plot(iterations, (am_loss - best_loss).mean(axis=0))
axs[1].fill_between(iterations,
                    np.quantile(am_loss - best_loss, 0.1, axis=0),
                    np.quantile(am_loss - best_loss, 0.9, axis=0),
                    alpha=0.2)

axs[1].plot(iterations, (ddl_loss - best_loss).mean(axis=0))
axs[1].fill_between(iterations,
                    np.quantile(ddl_loss - best_loss, 0.1, axis=0),
                    np.quantile(ddl_loss - best_loss, 0.9, axis=0),
                    alpha=0.2)

axs[1].plot(iterations[:len_ddl_steps+1], 
            (ddl_steps_loss[:, :len_ddl_steps+1] - best_loss).mean(axis=0))
axs[1].fill_between(iterations[:len_ddl_steps+1],
                    np.quantile(ddl_steps_loss[:, :len_ddl_steps+1] - best_loss, 0.1, axis=0),
                    np.quantile(ddl_steps_loss[:, :len_ddl_steps+1] - best_loss, 0.9, axis=0),
                    alpha=0.2)

axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_ylabel("log")
axs[1].set_ylabel(r"$F_N - F^*$")
axs[1].set_xlabel("Iterations N")
axs[1].set_title("Loss", fontsize=11)


# Score
am_recovery = np.zeros((len(results.keys()), len(iterations)))
ddl_recovery = np.zeros((len(results.keys()), len(iterations)))
ddl_steps_recovery = np.zeros((len(results.keys()), len(iterations)))

for j in results.keys():
    for i in range(len(iterations)):
        am_recovery[j, i] = recovery_score(results[j]["am"]["paths"][iterations[i]][-1], results[j]["dico"])
        ddl_recovery[j, i] = recovery_score(results[j]["ddl"]["paths"][iterations[i]][-1], results[j]["dico"])
        try:
            ddl_steps_recovery[j, i] = recovery_score(results[j]["ddl_steps"]["paths"][iterations[i]][-1], results[j]["dico"])
        except:
            ddl_steps_recovery[j, i] = 1

best_score = (np.min(np.concatenate([am_recovery.max(axis=1)[:, None], am_recovery.max(axis=1)[:, None]], axis=1), axis=1) + 1e-3)[:, None]

axs[2].plot(iterations, (best_score - am_recovery).mean(axis=0))
axs[2].fill_between(iterations,
                    np.quantile(best_score - am_recovery, 0.1, axis=0),
                    np.quantile(best_score - am_recovery, 0.9, axis=0),
                    alpha=0.2)

axs[2].plot(iterations, (best_score - ddl_recovery).mean(axis=0))
axs[2].fill_between(iterations,
                    np.quantile(best_score - ddl_recovery, 0.1, axis=0),
                    np.quantile(best_score - ddl_recovery, 0.9, axis=0),
                    alpha=0.2)

axs[2].plot(iterations[:len_ddl_steps+1],
            (best_score - ddl_steps_recovery[:, :len_ddl_steps+1]).mean(axis=0))
axs[2].fill_between(iterations[:len_ddl_steps+1],
                    np.quantile(best_score - ddl_steps_recovery[:, :len_ddl_steps+1],0.1, axis=0),
                    np.quantile(best_score - ddl_steps_recovery[:, :len_ddl_steps+1], 0.9, axis=0),
                    alpha=0.2)

axs[2].set_yscale("log")
axs[2].set_xscale("log")
axs[2].set_ylabel(r"$S_N - S^*$")
axs[2].set_xlabel("Iterations N")
axs[2].set_title("Rec. score", fontsize=11)

plt.tight_layout()
plt.savefig(str(FIGURES / "optim_path.pdf"))
