import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from tqdm import tqdm
from PIL import Image
from plipy.ddl import DDLInpaintingConv
from pathlib import Path

try:
    os.mkdir("../figures")
except OSError:
    pass

FIGURES = Path(__file__).resolve().parents[1] / "figures"
DATA = Path(__file__).resolve().parents[1] / "data"

im = Image.open(str(DATA / "flowers.png"))
im_gray = im.convert("L")
im_gray_resized = im_gray.resize((128, 128), Image.ANTIALIAS)

rho = 0

im_to_process = np.array(im_gray_resized) / 255.

omega = np.random.random(im_to_process.shape)
omega = (omega > rho).astype(float)
noise = np.random.normal(scale=0.1, size=im_to_process.shape)
im_noisy = np.clip(im_to_process + noise, 0, 1)


def draw_line():

    Ds = []

    for i in range(2):
        ddl = DDLInpaintingConv(100, 20, lambd=0.1, kernel_size=8,
                                learn_steps=False)
        ddl.fit(im_noisy[None, :, :], omega[None, :, :])
        D = ddl.get_prior()
        Ds.append(D)

    S1 = Ds[0]
    S2 = Ds[1]

    vs = 0.5 * (S2 - S1)
    alphas_synthesis = np.linspace(-3, 3, 100)
    score_line_synthesis = np.zeros(alphas_synthesis.shape)

    for i in range(alphas_synthesis.shape[0]):
        ddl.prior = torch.nn.Parameter(torch.tensor(S1 + vs + alphas_synthesis[i] * vs, dtype=torch.float, device=ddl.device))
        ddl.rescale()
        ddl.compute_lipschitz()
        score_line_synthesis[i] = ddl.cost(ddl.Y_tensor, ddl(ddl.Y_tensor))

    norm_synthesis = (score_line_synthesis - np.min(score_line_synthesis))
    norm_synthesis /= np.max(norm_synthesis)

    return alphas_synthesis, norm_synthesis


N_lines = 10

plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["mathtext.fontset"] = "cm"

plt.rc("text", usetex=False)
plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

plt.figure(figsize=(2.5, 2))

for i in tqdm(range(N_lines)):
    alphas_synthesis, norm_synthesis = draw_line()
    plt.plot(alphas_synthesis, norm_synthesis, alpha=0.3, color="blue")

plt.yticks([])
plt.xlabel("Normalized distance")
plt.title("CDL minima", fontsize=12)
plt.tight_layout()
plt.savefig(str(FIGURES / "shape_minima.pdf"))
