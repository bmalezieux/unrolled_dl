import numpy as np
import json

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from plipy.ddl import DDLInpaintingConv
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate2d

RESULTS = Path(__file__).resolve().parents[1] / "results"
NUM_SAMPLES = 50
RHOS = np.arange(0.1, 1., 0.1)
SNRS = np.arange(2., 21., 2.)


def cost_matrix(D, Dref):
    C = np.zeros((D.shape[0], Dref.shape[0]))
    for i in range(D.shape[0]):
        for j in range(Dref.shape[0]):
            C[i, j] = correlate2d(D[i, 0], Dref[j, 0]).max()
    return C


def recovery_score(D, Dref):
    """
    Comparison between a learnt prior and the truth
    """
    C = cost_matrix(D, Dref)

    row_ind, col_ind = linear_sum_assignment(C, maximize=True)
    score = C[row_ind, col_ind].mean()

    return score


def psnr(im, imref, d=1):
    mse = np.mean((im - imref)**2)
    return 10 * np.log10(d * d / mse)


def simul(image, snr=None):
    omega = np.random.random(image.shape)
    omega = (omega > 0).astype(float)
    if snr:
        var_signal = image.std() ** 2
        scale = 1. / (np.sqrt(10 ** (snr / 10.) / var_signal))
    else:
        scale = 0
    noise = np.random.normal(scale=scale, size=image.shape)
    im_noisy = np.clip(image + noise, 0, 1)

    ddl = DDLInpaintingConv(50, 20, lambd=0.1, kernel_size=8,
                            learn_steps=False)
    ddl.fit(im_noisy[None, :, :], omega[None, :, :], init="random")
    im_result_conv = np.clip(ddl.eval(), 0, 1)[0]

    return psnr(im_result_conv, image), ddl.get_prior()


path = Path(__file__).resolve().parents[1]
im = Image.open(str(path / "data/flowers.png"))
im_gray = im.convert("L")
im_gray_resized = im_gray.resize((128, 128), Image.ANTIALIAS)
im_to_process = np.array(im_gray_resized) / 255.

psnrs_synthesis = {}
scores_results = {}
for snr in tqdm(SNRS):
    psnrs_synthesis[snr] = []
    dicos = []
    for i in range(NUM_SAMPLES):
        psnr_result, dico = simul(im_to_process, snr)
        psnrs_synthesis[snr].append(psnr_result)
        dicos.append(dico)
    scores = np.zeros((NUM_SAMPLES, NUM_SAMPLES))
    for i in range(NUM_SAMPLES):
        scores[i, i] = 1
        for j in range(i+1, NUM_SAMPLES):
            scores[i, j] = recovery_score(dicos[i], dicos[j])
            scores[j, i] = scores[i, j]
    scores_results[snr] = scores

with open(str(RESULTS / 'psnrs_snr.json'), 'w') as filename:
    json.dump(psnrs_synthesis, filename)

with open(str(RESULTS / 'scores_snr.json'), 'w') as filename:
    json.dump(scores_results, filename)
