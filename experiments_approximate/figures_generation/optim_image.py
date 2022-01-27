import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


FIGURES = Path(__file__).resolve().parents[1] / "figures"
RESULTS = Path(__file__).resolve().parents[1] / "results"

iterations = np.load(str(RESULTS / 'image_iterations.npy'))
with open(str(RESULTS / 'optim_image.pickle'), 'rb') as file1:
    results = pickle.load(file1)

plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("text", usetex=False)
plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

plt.figure(figsize=(2.5, 2))

for key in results:
    if key != "DL":
        plt.plot(iterations, np.array(results[key]["psnr1"]), label=key)

plt.plot([0, 100],
         [results["DL"]["psnr1"], results["DL"]["psnr1"]], '--',
         label="DL-Oracle")

plt.legend(fontsize=8)
plt.xscale("log")
plt.xlabel("Iterations N")
plt.ylabel("PSNR")
plt.title("Denoising", fontsize=12)
plt.tight_layout()
plt.savefig(str(FIGURES / "optim_image.pdf"))
