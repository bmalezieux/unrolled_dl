import json
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

from pathlib import Path


try:
    os.mkdir("../figures")
except OSError:
    pass

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"


with open(str(RESULTS / "psnrs_snr.json")) as filename:
    results_psnrs = json.load(filename)


with open(str(RESULTS / "scores_snr.pickle"), 'rb') as filename:
    results_scores = pickle.load(filename)

tab_psnr = []
tab_scores = []
tab_snr = []

for key in results_psnrs:
    tab_snr.append(int(float(key)))
    tab_psnr.append(results_psnrs[key])

for key in results_scores:
    list_scores = []
    for i in range(results_scores[key].shape[0]):
        list_scores.append(results_scores[key][i].mean())
    tab_scores.append(list_scores)

plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("text", usetex=False)
plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

plt.figure(figsize=(2.5, 2))

plt.plot(tab_snr, np.array(tab_psnr).mean(axis=1), color="darkblue", linewidth=0.95)
plt.fill_between(tab_snr,
                 np.quantile(np.array(tab_psnr), 0, axis=1),
                 np.quantile(np.array(tab_psnr), 1, axis=1),
                 alpha=0.2,
                 color="darkblue")

labels = []
for i in range(len(tab_snr)):
    if i % 4 == 0:
        labels.append(tab_snr[i])
    else:
        labels.append("")
plt.ylabel("PSNR", color="darkblue")
# plt.xlabel("SNR (dB)")
# plt.tick_params(axis='y', labelcolor="darkblue")
plt.xticks(np.arange(len(tab_snr)) * 2, labels)
plt.xlabel("SNR (dB)")
plt.tick_params(axis='y', labelcolor="darkblue")
plt.yticks([21, 25])

ax2 = plt.twinx()
plt.plot(tab_snr, np.array(tab_scores).mean(axis=1), color="darkred", linewidth=0.95)
plt.fill_between(tab_snr,
                 np.quantile(np.array(tab_scores), 0, axis=1),
                 np.quantile(np.array(tab_scores), 1, axis=1),
                 alpha=0.2,
                 color="darkred")

ax2.set_ylabel("Rec. score", color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")
ax2.set_yticks([0.7, 0.9])
# ax2.set_ylabel("Rec. score", color="darkred")
# ax2.tick_params(axis="y", labelcolor="darkred")

# plt.figure(figsize=(2.5, 2))

# vp = plt.violinplot(tab_psnr,
#                     positions=np.arange(len(tab_psnr))*2-0.4)

# for partname in ('cbars', 'cmins', 'cmaxes'):
#     vp[partname].set_edgecolor("darkblue")
#     vp[partname].set_linewidth(1)

# for pc in vp["bodies"]:
#     pc.set_color("darkblue")

# plt.ylabel("PSNR", color="darkblue")


# labels = []
# for i in range(len(tab_snr)):
#     if i % 4 == 0:
#         labels.append(tab_snr[i])
#     else:
#         labels.append("")


# plt.xticks(np.arange(len(tab_snr)) * 2, labels)
# plt.xlabel("SNR (dB)")
# plt.tick_params(axis='y', labelcolor="darkblue")
# plt.yticks([21, 25])

# ax2 = plt.twinx()
# vp2 = ax2.violinplot(tab_scores,
#                      positions=np.arange(len(tab_psnr))*2+0.4)

# for partname in ('cbars', 'cmins', 'cmaxes'):
#     vp2[partname].set_edgecolor("darkred")
#     vp2[partname].set_linewidth(1)

# for pc in vp2["bodies"]:
#     pc.set_color("darkred")

# ax2.set_ylabel("Rec. score", color="darkred")
# ax2.tick_params(axis="y", labelcolor="darkred")
# ax2.set_yticks([0.7, 0.9])

plt.title("Min. distribution", fontsize=12)
plt.tight_layout()
plt.savefig(str(FIGURES / "scores_snr.pdf"))
plt.clf()
