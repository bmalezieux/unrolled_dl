import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from sklearn.feature_extraction.image import extract_patches_2d

from .base_prior_learning import BasePriorLearning
from .loss_and_gradients import (ista_iteration,
                                 fista_iteration,
                                 lasso)

from alphacsc.utils.dictionary import tukey_window


class DeepDictionaryLearning(BasePriorLearning):
    """
    Deep Dictionary Learning algorithm, based on (F)ISTA algorithm.
    Inherits from BasePriorLearning.

    Parameters
    ----------
    n_components : int
        Number of atoms in the prior.
    n_iter : int
        Number of unrolled iterations.
    lambd : float
        Regularization parameter.
        Default : 0.1.
    init_D : np.array, shape (width, height)
        Initialization for the prior.
        If None, the prior is initializaed from the data.
    device : str
        Device where the code is run ["cuda", "cpu"].
        If None, "cuda" is chosen if available.
    learn_steps : bool
        If True, the algorithm learns the step sizes after
        proper initialization of the prior.
        Default : True
    algo: str
        Algorithm to unroll ["ista", "fista"].
        Default : "fista".

    Attributes
    ----------
    device : str
        Device where the code is run ["cuda", "cpu"]
    lambd : float
        Regularization parameter
    n_iter : int
        Number of unrolled iterations
    learn_steps : bool
        If True, the algorithm learns the step sizes after
        proper initialization of the prior
    n_components : int
        Number of atoms in the prior
    dim_x : int
        Number of atoms
    dim_y : int
        Dimension of the measurements
    dim_signal : int
        Dimension of the signal
    init_D : np.array, shape (dim_signal, dim_x)
        Initialization for the prior
    lipschitz : float
        Lipschitz constant of the current prior
    operator : torch.Tensor, shape (dim_y, dim_signal)
        Measurement matrix
    prior : torch.Tensor, shape (dim_signal, dim_x)
        Current prior
    steps : torch.Tensor, shape (n_iter)
        steps sizes for the sparse coding algorithm
    Y_tensor : torch.Tensor, shape (dim_y, number of data)
        data
    algo: str
        Algorithm to unroll ["ista", "fista"].
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 init_D=None, device=None, learn_steps=True,
                 algo="fista"):

        super().__init__(n_components, n_iter, lambd,
                         init_D, device, learn_steps)

        self.algo = algo

    def rescale(self):
        """ Rescales columns """
        return super().rescale(atoms="columns")

    def compute_lipschitz(self):
        """
        Computes an upper bound of the Lipschitz
        constant of the gradient
        """
        with torch.no_grad():
            product = torch.matmul(self.operator, self.prior)
            self.lipschitz = torch.norm(
                torch.matmul(product, product.t())
            ).item()
            if self.lipschitz == 0:
                self.lipschitz = 1.

    def forward(self, y):
        """
        (F)ISTA-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (dim_y, number of data)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape (dim_x, number of data)
            Approximation of the sparse code associated to y
        """
        out = torch.zeros(
            (self.dim_x, y.shape[1]),
            dtype=torch.float,
            device=self.device
        )

        if self.algo == "fista":
            t = 1
            iterate = torch.zeros(
                (self.dim_x, y.shape[1]),
                dtype=torch.float,
                device=self.device
            )

        steps = self.steps / self.lipschitz
        product = torch.matmul(self.operator, self.prior)

        for i in range(self.n_iter):

            if self.algo == "fista":
                out, iterate, t = fista_iteration(out, iterate, y, product,
                                                  steps[i], t, self.lambd)
            elif self.algo == "ista":
                out = ista_iteration(out, y, product, steps[i], self.lambd)

        return out

    def cost(self, y, x):
        """ Lasso cost function """
        signal = torch.matmul(self.prior, x)
        return lasso(y, signal, x, self.operator, self.lambd)

    def training_process(self, backprop=True):
        """ Training process """
        return super().training_process(backprop)


class AMDictionaryLearning(DeepDictionaryLearning):
    """
    Dictionary Learning using alternating minimization.
    Inherits from DeepDictionaryLearning.
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 init_D=None, device=None, algo="fista"):
        learn_steps = False
        super().__init__(n_components, n_iter, lambd, init_D,
                         device, learn_steps, algo)

    def training_process(self):
        """ Training process """
        return super().training_process(backprop=False)


class DeepCDL(DeepDictionaryLearning):
    """
    Deep Convolutional Dictionary Learning for
    1D signals
    """
    def __init__(self, n_components, n_iter, lambd=0.1, kernel_size=5,
                 device=None, learn_steps=True, algo="fista"):
        super().__init__(n_components, n_iter, lambd, init_D=None,
                         device=device, learn_steps=learn_steps,
                         algo=algo)
        self.conv = torch.nn.functional.conv1d
        self.convt = torch.nn.functional.conv_transpose1d
        self.kernel_size = kernel_size

        self.window_tukey = torch.tensor(
            tukey_window(self.kernel_size),
            dtype=torch.float,
            device=self.device
        )[None, None, :]

    def compute_lipschitz(self):
        """ Compute the Lipschitz constant using the FFT"""
        with torch.no_grad():
            fourier_prior = fft.fft(self.prior, dim=2)
            self.lipschitz = torch.max(
                    torch.real(fourier_prior * torch.conj(fourier_prior)),
                    dim=2
            )[0].sum().item()
            if self.lipschitz == 0:
                self.lipschitz = 1

    def rescale(self):
        """
        Constrains the dictionary to have normalized atoms
        """
        with torch.no_grad():
            norm_col = torch.norm(self.prior, dim=(1, 2), keepdim=True)
            norm_col[torch.nonzero((norm_col == 0), as_tuple=False)] = 1
            self.prior /= norm_col
        return norm_col

    def get_prior(self):
        # window_dico = torch.zeros(
        #     self.prior.shape[2],
        #     device=self.device,
        #     dtype=torch.float)
        # window_dico[self.window_dico_len:-self.window_dico_len] = 1
        # D = self.prior * window_dico[None, None, :]
        D = self.prior * self.window_tukey

        return D.to("cpu").detach().numpy()

    def forward(self, y):
        """
        (F)ISTA-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (number of data, width, height)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape
            (number of data, n_components,
            width - kernel_size + 1)
            Approximation of the sparse code associated to y
        """

        out = torch.zeros(
            (y.shape[0],
             self.n_components,
             y.shape[1] - self.kernel_size + 1),
            dtype=torch.float,
            device=self.device
        )

        if self.algo == "fista":
            out_old = out.clone()
            t_old = 1

        steps = self.steps / self.lipschitz

        # window_dico = torch.zeros(
        #     self.prior.shape[2],
        #     device=self.device,
        #     dtype=torch.float)
        # window_dico[self.window_dico_len:-self.window_dico_len] = 1
        # D = self.prior * window_dico[None, None, :]

        D = self.prior * self.window_tukey

        for i in range(self.n_iter):
            # Gradient descent
            result1 = self.convt(out, D).sum(axis=1)
            result2 = self.conv(
                (result1 - y)[:, None, :],
                D
            )

            out = out - steps[i] * result2
            thresh = torch.abs(out) - steps[i] * self.lambd
            out = torch.sign(out) * F.relu(thresh)

            if self.algo == "fista":
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
                z = out + ((t_old-1) / t) * (out - out_old)
                out_old = out.clone()
                t_old = t
                out = z

        return out

    def cost(self, y, x):
        """ LASSO cost function """
        # window_dico = torch.zeros(
        #     self.prior.shape[2],
        #     device=self.device,
        #     dtype=torch.float)
        # window_dico[self.window_dico_len:-self.window_dico_len] = 1
        # D = self.prior * window_dico[None, None, :]
        D = self.prior * self.window_tukey
        signal = self.convt(x, D).sum(axis=1)
        res = signal - y
        l2 = (res * res).sum()
        l1 = torch.abs(x).sum()

        return 0.5 * l2 + self.lambd * l1

    def fit(self, data_y):
        """
        Training procedure

        Parameters
        ----------
        data_y : np.array, shape (number of data, width)
            Observations to be processed
        A : np.array, shape (number of data, width)
        init : str
            Method to initialize the dictionary ["patches", "random"].
            Default : "patches".

        Returns
        -------
        loss : float
        Final value of the loss after training.
        """

        self.dim_y = data_y.shape[0]

        # Dictionary
        self.prior = nn.Parameter(
            torch.rand(
                (self.n_components, 1, self.kernel_size),
                dtype=torch.float,
                device=self.device
            )
        )

        # Scaling and computing step
        self.rescale()
        self.steps = nn.Parameter(
            torch.ones(self.n_iter, device=self.device, dtype=torch.float)
        )
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(data_y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss


##############
# Inpainting #
##############


class DDLInpainting(DeepDictionaryLearning):
    """
    Deep Dictionary Learning based on (F)ISTA algorithm for inpainting.
    Inherits from DeepDictionaryLearning.
    """
    def __init__(self, n_components, n_iter, lambd=0.1, init_D=None,
                 device=None, learn_steps=True, algo="fista"):
        super().__init__(n_components, n_iter, lambd, init_D,
                         device, learn_steps, algo)

    def compute_lipschitz(self):
        """
        Computes an upper bound of the Lipschitz constant.
        In inpainting, the norm of the mask is equal to 1.
        """
        with torch.no_grad():
            self.lipschitz = torch.norm(
                torch.matmul(self.prior, self.prior.t())
            ).item()
            if self.lipschitz == 0:
                self.lipschitz = 1.

    def forward(self, y):
        """
        (F)ISTA-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (dim_y, number of data)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape (dim_x, number of data)
            Approximation of the sparse code associated to y
        """
        out = torch.zeros(
            (self.dim_x, y.shape[1]),
            dtype=torch.float,
            device=self.device
        )

        if self.algo == "fista":
            t = 1.
            iterate = torch.zeros((self.dim_x, y.shape[1]),
                                  dtype=torch.float,
                                  device=self.device)

        steps = self.steps / self.lipschitz

        for i in range(self.n_iter):

            if self.algo == "fista":
                out, iterate, t = fista_iteration(out, iterate, y,
                                                  self.operator, steps[i], t,
                                                  self.lambd, True, self.prior)
            elif self.algo == "ista":
                out = ista_iteration(out, y, self.operator, steps[i],
                                     self.lambd, True, self.prior)

        return out

    def cost(self, y, x):
        """ LASSO cost function for inpainting """
        signal = torch.matmul(self.prior, x)
        return lasso(y, signal, x, self.operator, self.lambd, True)

    def fit(self, data_y, A):
        """
        Training procedure

        Parameters
        ----------
        data_y : np.array, shape (dim_y, number of data)
            Observations to be processed
        A : np.array, shape (dim_y, number of data)
            Mask.
        """
        return super().fit(data_y, A)


class AMDLInpainting(DDLInpainting):
    """
    Dictionary Learning using alternating minimization.
    Inherits from DDLInpainting.
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 init_D=None, device=None, algo="fista"):
        learn_steps = False
        super().__init__(n_components, n_iter, lambd, init_D,
                         device, learn_steps, algo)

    def training_process(self):
        """ Training process """
        return super().training_process(backprop=False)


class DDLInpaintingConv(DeepDictionaryLearning):
    """
    Deep Convolutional Dictionary Learning based
    on (F)ISTA algorithm for inpainting.
    Inherits from DeepDictionaryLearning.

    Parameters
    ----------
    kernel_size : int
        Dimension of the square patch.
        Default : 5.

    Attributes
    ----------
    conv : torch.nn.functional.conv2d
        2D convolution operator
    convt : torch.nn.functional.conv_transpose2d
        2D correlation
    kernel_size : int
        Dimension of the square patch.
    """
    def __init__(self, n_components, n_iter, lambd=0.1, kernel_size=5,
                 device=None, learn_steps=True, algo="fista"):
        super().__init__(n_components, n_iter, lambd, None,
                         device, learn_steps, algo)

        self.conv = torch.nn.functional.conv2d
        self.convt = torch.nn.functional.conv_transpose2d
        self.kernel_size = kernel_size

    def compute_lipschitz(self):
        """ Compute the Lipschitz constant using the FFT """
        with torch.no_grad():
            fourier_prior = fft.fftn(self.prior, axis=(2, 3))
            self.lipschitz = torch.max(
                torch.max(
                    torch.real(fourier_prior * torch.conj(fourier_prior)),
                    dim=3
                )[0],
                dim=2
            )[0].sum().item()
            if self.lipschitz == 0:
                self.lipschitz = 1.

    def rescale(self):
        """
        Constrains the dictionary to have normalized columns

        Returns
        -------
        norm_col : torch.Tensor, shape (dim_x)
            Contains the norms of the current atoms.
        """
        with torch.no_grad():
            norm_col = torch.norm(self.prior, dim=(2, 3), keepdim=True)
            norm_col[torch.nonzero((norm_col == 0), as_tuple=False)] = 1
            self.prior /= norm_col
        return norm_col

    def forward(self, y):
        """
        (F)ISTA-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (number of data, width, height)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape
            (number of data, n_components,
            width - kernel_size + 1, height - kernel_size + 1)
            Approximation of the sparse code associated to y
        """
        out = torch.zeros(
            (y.shape[0],
             self.n_components,
             y.shape[1] - self.kernel_size + 1,
             y.shape[2] - self.kernel_size + 1),
            dtype=torch.float,
            device=self.device
        )

        if self.algo == "fista":
            out_old = out.clone()
            t_old = 1

        steps = self.steps / self.lipschitz

        for i in range(self.n_iter):
            # Gradient descent
            result1 = self.convt(out, self.prior).sum(axis=1)
            result2 = self.conv(
                (self.operator * (result1 - y))[:, None, :, :],
                self.prior
            )

            out = out - steps[i] * result2
            thresh = torch.abs(out) - steps[i] * self.lambd
            out = torch.sign(out) * F.relu(thresh)

            if self.algo == "fista":
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
                z = out + ((t_old-1) / t) * (out - out_old)
                out_old = out.clone()
                t_old = t
                out = z

        return out

    def cost(self, y, x):
        """ LASSO cost function """
        signal = self.convt(x, self.prior).sum(axis=1)
        return lasso(y, signal, x, self.operator, self.lambd, True)

    def fit(self, data_y, A, init="patches"):
        """
        Training procedure

        Parameters
        ----------
        data_y : np.array, shape (number of data, width, height)
            Observations to be processed
        A : np.array, shape (number of data, width, height)
            Mask.
        init : str
            Method to initialize the dictionary ["patches", "random"].
            Default : "patches".

        Returns
        -------
        loss : float
            Final value of the loss after training.
        """
        # Operator
        self.operator = torch.from_numpy(A).float().to(self.device)

        # Dictionary
        if init == "patches":
            d_init = extract_patches_2d(data_y[0],
                                        (self.kernel_size, self.kernel_size),
                                        max_patches=self.n_components)
            self.prior = nn.Parameter(
                torch.tensor(d_init[:, None, :, :],
                             dtype=torch.float,
                             device=self.device)
            )
        elif init == "random":
            self.prior = nn.Parameter(
                torch.rand(
                    (self.n_components, 1, self.kernel_size, self.kernel_size),
                    dtype=torch.float,
                    device=self.device
                )
            )

        # Scaling and computing step
        self.rescale()
        self.steps = nn.Parameter(
            torch.ones(self.n_iter, device=self.device, dtype=torch.float)
            )
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(data_y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss

    def eval(self):
        """
        Computes the results with one forward pass from the data

        Returns
        -------
        np.array
            Result image, after the forward pass and
            application of the dictionary.
        """
        with torch.no_grad():
            x = self.forward(self.Y_tensor)
            signal = self.convt(x, self.prior).sum(axis=1)
            return signal.to("cpu").detach().numpy()


class AMCDLInpainting(DDLInpaintingConv):
    """
    Convolutional Dictionary Learning using alternating minimization.
    Inherits from DDLInpaintingConv.
    """
    def __init__(self, n_components, n_iter, lambd=0.1, kernel_size=5,
                 init_D=None, device=None, algo="fista"):
        learn_steps = False
        super().__init__(n_components, n_iter, lambd, kernel_size,
                         init_D, device, learn_steps, algo)

    def training_process(self):
        """ Training process """
        return super().training_process(backprop=False)


###################
# Images in color #
###################


class DDLInpaintingConvColor(DDLInpaintingConv):
    """
    Deep Convolutional Dictionary Learning based
    on (F)ISTA algorithm for inpainting in color.
    Inherits from DDLInpaintingConv.
    """
    def __init__(self, n_components, n_iter, lambd=0.1, kernel_size=5,
                 device=None, learn_steps=True, algo="fista"):
        super().__init__(n_components, n_iter, lambd, kernel_size,
                         device, learn_steps, algo)

        self.conv = torch.nn.functional.conv3d
        self.convt = torch.nn.functional.conv_transpose3d

    def compute_lipschitz(self):
        """ Compute the Lipschitz constant using the FFT """
        with torch.no_grad():
            fourier_prior = fft.fftn(self.prior, axis=(2, 3, 4))
            self.lipschitz = torch.max(
                torch.max(
                    torch.real(fourier_prior * torch.conj(fourier_prior)),
                    dim=4
                )[0],
                dim=3
            )[0].sum().item()
            if self.lipschitz == 0:
                self.lipschitz = 1.

    def rescale(self):
        """
        Constrains the dictionary to have normalized columns

        Returns
        -------
        norm_col : torch.Tensor, shape (dim_x)
            Contains the norms of the current atoms.
        """
        with torch.no_grad():
            norm_col = torch.sqrt((self.prior ** 2).sum(axis=(2, 3, 4),
                                                        keepdim=True))
            # norm_col = torch.norm(self.prior, dim=[2, 3, 4], keepdim=True)
            norm_col[torch.nonzero((norm_col == 0), as_tuple=False)] = 1
            self.prior /= norm_col
        return norm_col

    def forward(self, y):
        """
        (F)ISTA-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (3, width, height)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape
            (number of data, n_components, 1,
            width - kernel_size + 1, height - kernel_size + 1)
            Approximation of the sparse code associated to y
        """
        out = torch.zeros(
            (y.shape[0],
             self.n_components,
             1,
             y.shape[2] - self.kernel_size + 1,
             y.shape[3] - self.kernel_size + 1),
            dtype=torch.float,
            device=self.device
        )

        if self.algo == "fista":
            out_old = out.clone()
            t_old = 1

        steps = self.steps / self.lipschitz

        for i in range(self.n_iter):
            # Gradient descent
            result1 = self.convt(out, self.prior).sum(axis=1)
            result2 = self.conv(
                (self.operator * (result1 - y))[:, None, :, :, :],
                self.prior
            )

            out = out - steps[i] * result2
            thresh = torch.abs(out) - steps[i] * self.lambd
            out = torch.sign(out) * F.relu(thresh)

            if self.algo == "fista":
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
                z = out + ((t_old-1) / t) * (out - out_old)
                out_old = out.clone()
                t_old = t
                out = z

        return out

    def fit(self, data_y, A, init="patches"):
        """
        Training procedure

        Parameters
        ----------
        data_y : np.array, shape (channels, width, height)
            Observations to be processed
        A : np.array, shape (channels, width, height)
            Mask.
        init : str
            Method to initialize the dictionary ["patches", "random"].
            Default : "patches".

        Returns
        -------
        loss : float
            Final value of the loss after training.
        """
        # Operator
        self.operator = torch.from_numpy(A).float().to(self.device)

        # Dictionary
        self.prior = nn.Parameter(
            torch.rand(
                (self.n_components, 1, 3, self.kernel_size, self.kernel_size),
                dtype=torch.float,
                device=self.device
            )
        )

        # Scaling and computing step
        self.rescale()
        self.steps = nn.Parameter(
            torch.ones(self.n_iter, device=self.device, dtype=torch.float)
        )
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(data_y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss
