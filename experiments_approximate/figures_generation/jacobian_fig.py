import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib import gridspec
from pathlib import Path

try:
    os.mkdir("../figures")
except OSError:
    pass


FIGURES = Path(__file__).resolve().parents[1] / "figures"
RESULTS = Path(__file__).resolve().parents[1] / "results"

plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("text", usetex=False)
plt.rc('font', family='serif', size=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

iterations = np.load(str(RESULTS / 'jac_iterations.npy'))
errors = np.load(str(RESULTS / 'jac_error.npy'))
distances = np.load(str(RESULTS / 'jac_distance_support.npy'))

iterations_backprop = np.load(str(RESULTS / 'jac_iterations_backprop.npy'))
iterations_2 = np.load(str(RESULTS / 'jac_iterations_2.npy'))
error_avg_K = np.load(str(RESULTS / 'jac_error_backprop.npy'))


# Figure from two samples


fig = plt.figure(figsize=(5.5, 1.3))
g = plt.GridSpec(2, 2, height_ratios=[0.25, 0.75], wspace=.4, hspace=.2, top=1, left=0, bottom=0, right=1)
g_jac = gridspec.GridSpecFromSubplotSpec(1, 2, g[1, 0], wspace=.6)
g_bp = gridspec.GridSpecFromSubplotSpec(1, 2, g[1, 1], wspace=.4)

label_jac = r'$\|J_l^N - J_l^*\|$'
label_sup = r'$\|S_N - S^*\|_0$'

colors = {
    'jac': 'darkred',
    'sup': 'darkblue',
    200: 'C0',
    50: 'C1',
    20: 'C2'
}

for id_ax, id_atom in enumerate([0, 21]):

    # Plot Jacobian convergence
    color = 'darkred'
    ax = fig.add_subplot(g_jac[:, id_ax])
    ax.set_xlabel('Iterations N')
    ax.set_yscale("log")
    ax.set_xscale("symlog")
    ax.plot(iterations, errors[id_atom], color=colors['jac'])
    ax.tick_params(axis='y', labelcolor=colors['jac'])
    # if id_ax == 0:
    #     ax.set_ylabel(label_jac, color=colors['jac'])
    ax.set_xticks([1, 100, 10000])

    # Plot Support convergence
    ax2 = ax.twinx()
    # if id_ax == 1:
    #     ax2.set_ylabel('Dist. from support', color=color)
    #     ax2.set_ylabel(label_sup, color=colors['sup'])
    ax2.plot(distances[id_atom], color=colors['sup'], label=label_sup)
    ax2.tick_params(axis='y', labelcolor=colors['sup'])

    # Plot different truncated backprop
    ax = fig.add_subplot(g_bp[:, id_ax])
    ax.plot(
        iterations_2, errors[id_atom][iterations_2], label="full",
        color=colors['jac']
    )
    for j in range(error_avg_K.shape[1]):
        K = iterations_backprop[j]
        ax.plot(
            iterations_2, error_avg_K[id_atom, j, :], label=str(K),
            color=colors[K]
        )

    ax.set_xlabel("Iterations N")
    if id_ax == 0:
        ax.set_ylabel(label_jac)
    ax.tick_params(axis='y')
    ax.set_xscale("symlog")
    ax.set_xticks([1, 100, 10000])
    legend = ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    legend.remove()

ax_legend_bp = fig.add_subplot(g[0, 1])
ax_legend_bp.set_axis_off()
legend = ax_legend_bp.legend(
    handles, labels, loc='center', ncol=4, title="Max BP depth",
    columnspacing=0.8, handlelength=1.2, handletextpad=.4,
)
legend._legend_box.align = "left"


ax_legend_jac = fig.add_subplot(g[0, 0])
ax_legend_jac.set_axis_off()
labels = [label_jac, label_sup]
handles = [
    plt.Line2D([], [], color=colors['jac']),
    plt.Line2D([], [], color=colors['sup']),
]
legend = ax_legend_jac.legend(
    handles, labels, loc='center', ncol=2,
    columnspacing=0.8, handlelength=1.2, handletextpad=.4,
)
legend._legend_box.align = "left"

plt.savefig(str(FIGURES / "conv_jac.pdf"))
plt.clf()


# Figure presenting all samples

plt.figure(figsize=(1.375, 1.375))

cpt = 0
cpt_errors = 0
for i in range(len(errors)):
    if errors[i].max() > errors[i][0]:
        plt.plot(iterations, errors[i], alpha=0.2, c='r')
        cpt_errors += 1
    else:
        plt.plot(iterations, errors[i], alpha=0.2, c='k')
    cpt += 1

# print(cpt, cpt_errors, cpt_errors / cpt)

plt.xscale("log")
plt.xlabel("Iterations N", fontsize=9)
plt.ylabel(label_jac, fontsize=9)

plt.savefig(str(FIGURES / "conv_jac_general.pdf"))
plt.clf()
