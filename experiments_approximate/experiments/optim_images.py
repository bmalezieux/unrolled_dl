import numpy as np
import pickle
import time

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from utils import (create_patches_overlap,
                   patch_average)
from plipy.ddl import DeepDictionaryLearning, AMDictionaryLearning


RESULTS = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(2022)
SIGMA = 0.1


class FISTA():
    """
    FISTA algorithm
    """
    def __init__(self, y, D, lambd=1):
        self.y = y
        self.D = D

        # Regularization term
        self.lambd = lambd

        # Lipschitz constant of the gradient
        self.lipschitz = self.compute_lipschitz()

        # Step
        self.gamma = 1 / self.lipschitz

    def compute_lipschitz(self, eps=1e-5):
        """
        Computes the lispchitz constant of the gradient
        with power iteration.

        Parameters
        ----------
        eps : float
            Convergence tolerance.

        Returns
        -------
        norm : float
            Norm of the gradient.
        """
        x_random = RNG.random(self.D.shape[1])
        x_random /= np.linalg.norm(x_random)
        old_norm = 0.
        norm = 1.
        while np.abs(old_norm - norm) > eps:
            old_norm = norm
            x_random = self.D.T @ self.D @ x_random
            norm = np.linalg.norm(x_random)
            x_random /= norm

        return norm

    def f(self, x):
        """ Data fitting term of the loss """
        res = self.D @ x - self.y
        res *= res
        return res.sum(axis=0) / 2

    def g(self, x):
        """ L1 norm """
        return np.linalg.norm(x, ord=1, axis=0)

    def energy(self, x):
        """ Lasso cost function """
        return self.f(x) + self.lambd * self.g(x)

    def grad_f(self, x):
        """ Gradient of data fitting term """
        return self.D.T @ (self.D @ x - self.y)

    def prox_g(self, x, gamma):
        """ Soft thresholding """
        return np.sign(x) * np.maximum(0, np.abs(x) - gamma * self.lambd)

    def gradientDescent(self, x, gamma):
        """ Proximal gradient descent """
        return self.prox_g(x - gamma * self.grad_f(x), gamma)

    def iter(self, n_iter):
        """ FISTA algorithm with n_iter iterations """
        x = self.D.T @ np.zeros(self.y.shape)
        z = x.copy()
        xold = x.copy()
        told = 1
        for i in range(n_iter):
            x = self.gradientDescent(z, self.gamma)
            t = 0.5 * (1 + np.sqrt(1 + 4 * told*told))
            z = x + ((told-1) / t) * (x - xold)
            told = t
            xold = x.copy()
        return x


class ImageDenoising():
    """
    Image denoising with dictionary learning
    """
    def __init__(self, y, n_iter=20, lambd=1, m=12,
                 n_components=None, algo="fista"):
        self.m = m
        self.r, self.c = y.shape
        self.y, _ = create_patches_overlap(y, m)
        self.lambd = lambd
        self.n_iter = n_iter
        self.D = None

        if n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = 2 * m * m

        self.algo = algo

    def get_prior(self):
        return self.D

    def training_process(self, grad=True, steps=True):
        if not grad:
            ddl = AMDictionaryLearning(self.n_components,
                                       self.n_iter,
                                       lambd=self.lambd)
            loss = ddl.fit(self.y)
        elif steps:
            ddl = DeepDictionaryLearning(self.n_components,
                                         self.n_iter,
                                         lambd=self.lambd)
            loss = ddl.fit(self.y)
        else:
            ddl = DeepDictionaryLearning(self.n_components,
                                         self.n_iter,
                                         lambd=self.lambd,
                                         learn_steps=steps)
            loss = ddl.fit(self.y)

        self.D = ddl.get_prior()
        result1 = ddl.eval()
        sparse_coding = FISTA(self.y, self.D, lambd=self.lambd)
        result2 = sparse_coding.iter(1000)
        im_result1 = patch_average(self.D @ result1, self.m, self.r, self.c)
        im_result2 = patch_average(self.D @ result2, self.m, self.r, self.c)
        return im_result1, im_result2, loss


path = Path(__file__).resolve().parents[1]
im = Image.open(str(path / "data/flowers.png"))
im_gray = im.convert("L")
im_gray_resized = im_gray.resize((128, 128), Image.ANTIALIAS)
im_to_process = np.array(im_gray_resized) / 255.

noise = RNG.normal(scale=SIGMA, size=im_to_process.shape)
im_noisy = np.clip(im_to_process + noise, 0, 1)


# Optimization
lambd = 0.1
n_components = 128
m_patch = 10


def psnr(im, imref, d=1):
    mse = np.mean((im - imref)**2)
    return 10 * np.log(d * d / mse) / np.log(10)


denoiser = ImageDenoising(im_noisy, lambd=lambd,
                          m=m_patch, n_components=n_components)


iterations = np.unique(np.logspace(0, np.log10(100), num=50, dtype=int))
results = {
    "AM": {"psnr1": [], "psnr2": [], "loss": [], "time": []},
    "DDL": {"psnr1": [], "psnr2": [], "loss": [], "time": []},
    "DDL_steps": {"psnr1": [], "psnr2": [], "loss": [], "time": []},
    "DL": {"psnr1": [], "psnr2": [], "loss": [], "time": []}
    }


for i in tqdm(iterations):
    denoiser.n_iter = i
    start = time.time()
    im1_ddl, im2_ddl, loss_ddl = denoiser.training_process(grad=True,
                                                           steps=False)
    stop = time.time()
    results["DDL"]["psnr1"].append(psnr(im1_ddl, im_to_process))
    results["DDL"]["psnr2"].append(psnr(im2_ddl, im_to_process))
    results["DDL"]["loss"].append(loss_ddl)
    results["DDL"]["time"].append(stop - start)

    start = time.time()
    im1_ddl_steps, im2_ddl_steps, loss_ddl_steps = denoiser.training_process(grad=True,
                                                                             steps=True)
    stop = time.time()
    results["DDL_steps"]["psnr1"].append(psnr(im1_ddl_steps, im_to_process))
    results["DDL_steps"]["psnr2"].append(psnr(im2_ddl_steps, im_to_process))
    results["DDL_steps"]["loss"].append(loss_ddl_steps)
    results["DDL_steps"]["time"].append(stop - start)

    start = time.time()
    im1_am, im2_am, loss_am = denoiser.training_process(grad=False,
                                                        steps=False)
    stop = time.time()
    results["AM"]["psnr1"].append(psnr(im1_am, im_to_process))
    results["AM"]["psnr2"].append(psnr(im2_am, im_to_process))
    results["AM"]["loss"].append(loss_am)
    results["AM"]["time"].append(stop - start)


start = time.time()
denoiser.n_iter = 1000
im1_am, im2_am, loss_am = denoiser.training_process(grad=False,
                                                    steps=False)

stop = time.time()
results["DL"]["psnr1"].append(psnr(im1_am, im_to_process))
results["DL"]["psnr2"].append(psnr(im2_am, im_to_process))
results["DL"]["loss"].append(loss_am)
results["DL"]["time"].append(stop - start)


np.save(str(RESULTS / "image_iterations.npy"), iterations)

with open(str(RESULTS / 'optim_image.pickle'), 'wb') as file1:
    pickle.dump(results, file1)
