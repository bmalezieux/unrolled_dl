import numpy as np
import matplotlib.pyplot as plt
import pickle

from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"

with open(str(RESULTS / 'optim_online_am_results.pickle'), 'rb') as file1:
    results_am = pickle.load(file1)

with open(str(RESULTS / 'optim_online_ddl_results.pickle'), 'rb') as file1:
    results_ddl = pickle.load(file1)

lambdas = np.load(str(RESULTS / "optim_online_lambdas.npy"))
times_ddl = []
times_am = []

for i in range(len(lambdas)):
    times_ddl.append([])
    times_am.append([])
    for key in results_ddl:
        if lambdas[i] in results_ddl[key]:
            times_ddl[i].append(results_ddl[key][lambdas[i]])
        else:
            times_ddl[i].append(200)
    for key in results_am:
        if lambdas[i] in results_am[key]:
            times_am[i].append(results_am[key][lambdas[i]])
        else:
            times_am[i].append(200)


plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("text", usetex=False)
plt.rc('font', family='serif', size=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

plt.figure(figsize=(3, 2))

plt.plot(lambdas, np.array(times_ddl).mean(axis=1), label="Sto. DDL")
plt.fill_between(
    lambdas,
    np.quantile(np.array(times_ddl), 0.1, axis=1),
    np.quantile(np.array(times_ddl), 0.9, axis=1),
    alpha=0.2
    )
plt.plot(lambdas, np.array(times_am).mean(axis=1), label="Online DL")
plt.fill_between(
    lambdas,
    np.quantile(np.array(times_am), 0.1, axis=1),
    np.quantile(np.array(times_am), 0.9, axis=1),
    alpha=0.2
    )
plt.legend(fontsize=9)
plt.title("Time to reach a score of 0.95", fontsize=10)
plt.xlabel(r"$\lambda$")
plt.ylabel("Time (s)")
plt.ylim([5, 150])
plt.xlim([0.1, 1])
plt.yscale("log")
plt.tight_layout()
plt.savefig(str(FIGURES / "online_lambda_time.pdf"))
