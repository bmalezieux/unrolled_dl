import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

try:
    os.mkdir("../figures")
except OSError:
    pass


FIGURES = Path(__file__).resolve().parents[1] / "figures"
RESULTS = Path(__file__).resolve().parents[1] / "results"


iterations = np.load(str(RESULTS / 'angles_iterations.npy'))
iterations_backprop = np.load(str(RESULTS / 'angles_iterations_backprop.npy'))
angles = np.load(str(RESULTS / 'angles_g1_g2.npy'))
norms = np.load(str(RESULTS / 'diff_g1_g2.npy'))

iterations_image = np.load(str(RESULTS / 'angles_iterations_image.npy'))
iterations_backprop_image = np.load(
    str(RESULTS / 'angles_iterations_backprop_image.npy')
    )
angles_image = np.load(str(RESULTS / 'angles_g1_g2_image.npy'))
norms_image = np.load(str(RESULTS / 'diff_g1_g2_image.npy'))


plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("text", usetex=False)
plt.rc('font', family='serif', size=11)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)

# figsize=(6.875, 1.875)
fig, axs = plt.subplots(1, 3, figsize=(6.875, 1.875))
lines = []
lines_label = []

lines.append(axs[0].plot(iterations, 1 - angles[:, 0])[0])
lines_label.append("AM")

for i in range(1, len(iterations_backprop) - 1):
    lines.append(axs[0].plot(iterations, 1 - angles[:, i])[0])
    lines_label.append(str(iterations_backprop[i]))

lines.append(axs[0].plot(iterations, 1 - angles[:, -1])[0])
lines_label.append("full")


axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel("Iterations N")
axs[0].set_ylabel(r"$1 - \langle g, g^* \rangle$")
axs[0].set_ylim([1e-8, 1])
axs[0].set_title("Gaussian dictionary", fontsize=11)

axs[1].plot(iterations_image, 1 - angles_image[:, 0])

for i in range(1, len(iterations_backprop) - 1):
    axs[1].plot(iterations_image, 1 - angles_image[:, i])

axs[1].plot(iterations_image, 1 - angles_image[:, -1])


axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlabel("Iterations N")
axs[1].set_ylabel(r"$1 - \langle g, g^* \rangle$")
axs[1].set_title("Noisy image", fontsize=11)


axs[2].plot([0, 1000], [0, 0])
for i in range(1, len(iterations_backprop) - 1):
    axs[2].plot(iterations_image,
                (angles_image[:, i] - angles_image[:, 0]) / (1 - angles_image[:, 0]))

axs[2].plot(iterations_image,
            (angles_image[:, -1] - angles_image[:, 0]) / (1 - angles_image[:, 0]))


axs[2].set_xscale("log")
axs[2].set_ylim([-0.3, 0.3])
axs[2].set_xlabel("Iterations N")
axs[2].set_ylabel("Relative diff.")
axs[2].set_title("Noisy image", fontsize=11)


fig.legend(lines, lines_label, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.3), title="BP depth")
plt.tight_layout()
plt.savefig(str(FIGURES / "gradients_angle.pdf"))
plt.clf()


fig = plt.figure(figsize=(6, 2))

gs = plt.GridSpec(1, 2, width_ratios=[10, 10])


axi = fig.add_subplot(gs[0, 0])
for i in range(1, len(iterations_backprop) - 1):
    axi.plot(iterations, norms[:, i], label=str(iterations_backprop[i]))

axi.plot(iterations, norms[:, -1], label="full")
axi.plot(iterations, norms[:, 0], label="AM")
axi.set_xscale("log")
axi.set_yscale("log")
axi.set_xlabel("Iterations N")
axi.set_ylabel(r"$||g^* - g||$")
axi.set_title("Gaussian dictionary", fontsize=11)
axi.legend(fontsize=7, title_fontsize=7, title="BP depth")


axi = fig.add_subplot(gs[0, 1])
for i in range(1, len(iterations_backprop) - 1):
    axi.plot(iterations, norms_image[:, i], label=str(iterations_backprop[i]))

axi.plot(iterations, norms_image[:, -1])
axi.plot(iterations, norms_image[:, 0])
axi.set_xscale("log")
axi.set_yscale("log")
axi.set_xlabel("Iterations N")
axi.set_ylabel(r"$||g^* - g||$")
axi.set_title("Noisy image", fontsize=11)


plt.tight_layout()
plt.savefig(str(FIGURES / "gradients_norms.pdf"))
plt.clf()