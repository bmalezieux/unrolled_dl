import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from utils import create_patches_overlap
from tqdm import tqdm


RESULTS = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(2022)


def compute_lipschitz(D):
    """
    Power iteration to compute the Lipschitz constant
    of the gradient of the data fitting term

    Parameters
    ----------
    D : np.array
        dictionary

    Returns
    -------
    float
        Lipschitz constant
    """
    iterations = 50
    u = RNG.random(D.shape[1])
    for i in range(iterations):
        u = D.T @ D @ u
        norme = np.linalg.norm(u)
        u /= norme
    return norme


def iter_algo(y, D, n_iter, L, algo="fista", lambd=0.1):
    out = np.zeros((D.shape[1], y.shape[1]))

    if algo == "fista":
        out_old = out.copy()
        t_old = 1.

    step = 1. / L

    for i in range(n_iter):

        # Gradient descent
        out = out - step * D.T @ (D @ out - y)

        # Thresholding
        thresh = np.abs(out) - step * lambd
        out = np.sign(out) * np.maximum(0, thresh)

        if algo == "fista":
            t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
            z = out + ((t_old-1) / t) * (out - out_old)
            out_old = out.copy()
            t_old = t
            out = z

    return out


class DeepISTA(nn.Module):
    def __init__(self, n_iter, D, L, algo="fista", lambd=0.1,
                 device=None, n_iter_backprop=None):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Regularization parameter
        self.lambd = lambd

        # Algorithm unrolled and parameters
        self.algo = algo
        self.n_iter = n_iter

        if n_iter_backprop is None:
            self.backprop_iter = n_iter
        else:
            self.backprop_iter = n_iter_backprop

        # Shape
        self.dim_x = D.shape[1]

        # Initial dictionary
        self.init_D = D
        self.lipschitz = L

    def forward(self, y):
        """
        (F)ISTA-like forward pass
        """
        out = torch.zeros((self.dim_x, y.shape[1]),
                          dtype=torch.float,
                          device=self.device)

        if self.algo == "fista":
            out_old = out.clone()
            t_old = 1

        step = 1. / self.lipschitz

        torch.set_grad_enabled(False)

        for i in range(self.n_iter):
            if i >= self.n_iter - self.backprop_iter:
                torch.set_grad_enabled(True)
            # Gradient descent
            out = out - step * torch.matmul(
                self.dictionary.t(),
                torch.matmul(self.dictionary, out) - y
                )
            thresh = torch.abs(out) - step * self.lambd
            out = torch.sign(out) * F.relu(thresh)

            if self.algo == "fista":
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
                z = out + ((t_old-1) / t) * (out - out_old)
                out_old = out.clone()
                t_old = t
                out = z

        return out

    def cost(self, y, x):
        """
        LASSO cost function
        """
        signal = torch.matmul(self.dictionary, x)
        res = signal - y
        l2 = (res * res).sum()
        l1 = torch.abs(x).sum()

        return 0.5 * l2 + self.lambd * l1

    def compute_gradient(self, data_y):
        """
        Training procedure
        """
        # Dimension
        self.dim_y = data_y.shape[0]

        # Dictionary
        dico_tensor = torch.from_numpy(self.init_D).float().to(self.device)
        self.dictionary = nn.Parameter(dico_tensor)

        # Data
        Y_tensor = torch.from_numpy(data_y).float().to(self.device)

        if self.backprop_iter == 0:
            with torch.no_grad():
                out = self.forward(Y_tensor)
        else:
            out = self.forward(Y_tensor)
        loss = self.cost(Y_tensor, out)
        loss.backward()

        grad = self.dictionary.grad.to("cpu").detach().numpy()
        self.dictionary.grad.zero_()

        return grad


def compute_angles_dico(data, D, n_iter, lambd, algo,
                        iterations, iterations_backprop):

    lipschitz = compute_lipschitz(D)
    x_result = iter_algo(data, D, n_iter, lipschitz,
                         algo=algo, lambd=lambd)
    dista = DeepISTA(n_iter=20, D=D, L=lipschitz, lambd=lambd, algo=algo)
    g_star = (D @ x_result - data) @ x_result.T

    angles = np.zeros((len(iterations), len(iterations_backprop)))
    norms = np.zeros((len(iterations), len(iterations_backprop)))

    for i in tqdm(range(len(iterations))):
        for j in range(len(iterations_backprop)):
            dista.n_iter = iterations[i]
            dista.backprop_iter = min(iterations_backprop[j], 100)
            g2 = dista.compute_gradient(data)
            den = np.sqrt(np.trace(g2.T @ g2) * np.trace(g_star.T @ g_star))
            angle2 = np.trace(g2.T @ g_star) / den
            angles[i, j] = angle2
            norms[i, j] = np.linalg.norm(g2 - g_star)

    return angles, norms


def generate_data(dico, N, k=0.3):
    """
    Generate data from dictionary

    Parameters
    ----------
    dico : np.array
        dictionary
    N : int
        number of samples
    k : float, optional
        sparsity, by default 0.3

    Returns
    -------
    (np.array, np.array)
        signal, sparse codes
    """
    d = dico.shape[1]
    X = (RNG.random((d, N)) > (1-k)).astype(float)
    X *= RNG.normal(scale=1, size=(d, N))
    return dico @ X, X


print("Synthetic data")
# Synthetic data
d = 50
P = 30
N = 1000
sigma = 0.1

A = RNG.normal(size=(P, d)) / P
A /= np.sqrt(np.sum(A**2, axis=0))

data_x, codes = generate_data(A, N)

noise = RNG.normal(scale=sigma, size=data_x.shape)
data = data_x + noise


# Optimization parameters
lambd = 0.1
sigma_dic = A.std() * 0.5
current_dictionary = A + RNG.normal(scale=sigma_dic, size=A.shape)


n_iter = 10000
algo = "fista"
iterations = np.unique(np.logspace(0, 3, dtype=int, num=100))
iterations_backprop = [0, 20, 50, 1000]

angles, norms = compute_angles_dico(data, current_dictionary,
                                    n_iter, lambd, algo, iterations,
                                    iterations_backprop)

np.save(str(RESULTS / "angles_iterations.npy"), iterations)
np.save(
    str(RESULTS / "angles_iterations_backprop.npy"),
    iterations_backprop
    )
np.save(str(RESULTS / "angles_g1_g2.npy"), angles)
np.save(str(RESULTS / "diff_g1_g2.npy"), norms)


print("Image data")
# Image
M_PATCH = 10
COMPONENTS = 128
sigma = 0.1

path = Path(__file__).resolve().parents[1]
im = Image.open(str(path / "data/flowers.png"))
im_gray = im.convert("L")
im_gray_resized = im_gray.resize((128, 128), Image.ANTIALIAS)
im_to_process = np.array(im_gray_resized) / 255.

noise = RNG.normal(scale=sigma, size=im_to_process.shape)
im_noisy = im_to_process + noise

data, _ = create_patches_overlap(im_noisy, M_PATCH)


# Optimization image
lambd = 0.1
current_dictionary = data[:, :COMPONENTS]

n_patches = 1000
n_iter = 10000
algo = "fista"
iterations = np.unique(np.logspace(0, 3, dtype=int, num=100))
iterations_backprop = [0, 20, 50, 1000]

angles, norms = compute_angles_dico(data[:, :n_patches], current_dictionary,
                                    n_iter, lambd, algo, iterations,
                                    iterations_backprop)


np.save(str(RESULTS / "angles_iterations_image.npy"), iterations)
np.save(
    str(RESULTS / "angles_iterations_backprop_image.npy"),
    iterations_backprop
    )
np.save(str(RESULTS / "angles_g1_g2_image.npy"), angles)
np.save(str(RESULTS / "diff_g1_g2_image.npy"), norms)
