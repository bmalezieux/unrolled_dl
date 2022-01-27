import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft

from .base_prior_learning import BasePriorLearning
from .loss_and_gradients import lasso, condat_vu_iteration


class DeepPrimalDualPriorLearning(BasePriorLearning):
    """
    Analysis prior learning, based on Condat-Vu algorithm,
    with UN constraint.
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
    taus : torch.Tensor
        Step sizes.
    steps : torch.Tensor
        Step sizes.
    constraint : str
        Determines the initial prior ["un", "untf", "conv"].
    kernel_size : int
        Size of the convolutional kernel for constraint set to "conv".
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 init_D=None, device=None, learn_steps=True):
        super().__init__(n_components, n_iter, lambd,
                         init_D, device, learn_steps)

        # Steps sizes for Condat-Vu
        self.taus = None
        self.steps = None

        # Constraint
        self.constraint = "un"

        # Kernel size for convolutions
        self.kernel_size = None

    def rescale(self):
        """ Rescales rows """
        return super().rescale(atoms="rows")

    def replace_zeros(self):
        """ Replaces null rows with random rows for UNTF """
        with torch.no_grad():
            sums = torch.sum(torch.abs(self.prior), axis=1)
            positions = torch.nonzero((sums == 0), as_tuple=False)
            self.prior.data[positions] = torch.rand(
                size=self.prior[positions].size(),
                dtype=torch.float,
                device=self.device
                )

    def proj_tf(self):
        """ Projection on tight frames priors for UNTF """
        with torch.no_grad():
            U, _, V = torch.svd(self.prior)
            first_mul = torch.matmul(U, torch.eye(U.shape[1], V.t().shape[0],
                                                  dtype=torch.float,
                                                  device=self.device))
            self.prior.data = torch.matmul(first_mul, V.t())

    def forward(self, y):
        """
        Condat-Vu-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (dim_y, number of data)
            Data to be processed by (F)ISTA

        Returns
        -------
        out_p : torch.Tensor, shape (dim_x, number of data)
            Approximation of the signal associated to y
        """

        out_p = torch.zeros((self.dim_x, y.shape[1]),
                            dtype=torch.float,
                            device=self.device)

        out_d = torch.zeros((self.prior.shape[0], y.shape[1]),
                            dtype=torch.float,
                            device=self.device)

        self.taus = 1 / (0.5 * self.norm_A + self.steps * self.lipschitz)

        for i in range(self.n_iter-1):

            out_p, out_d = condat_vu_iteration(
                out_p,
                out_d,
                y,
                self.operator,
                self.prior,
                self.steps[i],
                self.taus[i],
                self.lambd
            )

        # Last iteration
        # Only out_p is updated, the full iteration is not called
        # to prevent issues linked to null gradients.
        first_gradient = torch.matmul(
            self.operator.t(), torch.matmul(self.operator, out_p) - y)
        second_gradient = torch.matmul(self.prior.t(), out_d)
        out_p = out_p - self.taus[-1] * (first_gradient + second_gradient)

        return out_p

    def cost(self, y, x):
        """ LASSO cost function """
        z = torch.matmul(self.prior, x)
        return lasso(y, x, z, self.operator, self.lambd)

    def fit(self, data_y, A=None):
        """
        Training procedure

        Parameters
        ----------
        data_y : np.array, shape (dim_y, number of data)
            Observations to be processed.
        A : np.array, shape (dim_y, dim_signal)
            Measurement matrix.
            If set to None, A is considered to be the identity.
            Default : None.

        Returns
        -------
        loss : float
            Final value of the loss after training.
        """
        # Dimension
        self.dim_y = data_y.shape[0]

        if A is None:
            self.dim_signal = self.dim_y
        else:
            self.dim_signal = A.shape[1]

        # Operator
        if A is None:
            self.operator = torch.eye(
                self.dim_y, device=self.device, dtype=torch.float
                )
        else:
            self.operator = torch.from_numpy(A).float().to(self.device)

        self.dim_x = self.operator.shape[1]

        # Prior
        if self.init_D is None:
            choice = np.random.choice(data_y.shape[1], self.n_components)
            dico = data_y[:, choice]
            self.prior = nn.Parameter(
                torch.tensor(dico, device=self.device, dtype=torch.float)
                )
        else:
            dico_tensor = torch.from_numpy(self.init_D).float().to(self.device)
            self.prior = nn.Parameter(dico_tensor)

        if self.constraint == "conv":
            # Conv prior
            self.prior = nn.Parameter(
                torch.rand((self.n_components, 1, self.kernel_size),
                           dtype=torch.float,
                           device=self.device)
                )

        # Scaling and computing step
        if self.constraint == "untf":
            self.replace_zeros()
            self.proj_tf()
        self.rescale()
        self.steps = nn.Parameter(
            torch.ones(self.n_iter, device=self.device, dtype=torch.float)
            )
        self.taus = torch.zeros_like(self.steps)

        self.norm_A = torch.norm(
            torch.matmul(self.operator, self.operator.t())
            ).item()
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(data_y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss


class DeepPrimalDualPriorLearningUNTF(DeepPrimalDualPriorLearning):
    """
    Analysis prior learning with UNTF constraint.
    Inherits from DeepPrimalDualPriorLearning.
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 init_D=None, device=None, learn_steps=True):
        super().__init__(n_components, n_iter, lambd, init_D,
                         device, learn_steps)

        self.constraint = "untf"

    def line_search(self, step, loss, state):
        """
        Line search and gradient descent.
        A Cayley transform is used to optimize on the Stiefel manifold.
        """
        # Line search parameters
        beta = 0.5

        # Learning rate
        t = step

        ok = False
        end = False

        with torch.no_grad():
            A = torch.matmul(self.prior.grad, self.prior.t()) -\
                torch.matmul(self.prior, self.prior.grad.t())
            Id = torch.eye(A.shape[0], dtype=torch.float, device=self.device)
            prior_copy = self.prior.data.clone()

            # Learning step

            if state:
                sigmas_copy = self.steps.data.clone()
                grad_sigmas_copy = self.steps.grad.data.clone()

                self.steps.data = sigmas_copy - t * grad_sigmas_copy

            tau = t / 2.
            Q = torch.matmul(torch.inverse(Id + tau * A),
                             Id - tau * A)
            self.prior.data = torch.matmul(Q, prior_copy)

            init = True

            while not ok:
                if not init:
                    # Backtracking
                    tau = t / 2.
                    Q = torch.matmul(torch.inverse(Id + tau * A),
                                     Id - tau * A)
                    self.prior.data = torch.matmul(Q, prior_copy)

                    if state:
                        self.steps.data = sigmas_copy - t * grad_sigmas_copy
                else:
                    init = False

                # Projection
                self.rescale()

                # Computing lipschitz constant
                self.compute_lipschitz()

                # Computing loss with new parameters
                current_cost = self.cost(self.Y_tensor,
                                         self.forward(self.Y_tensor)).item()

                if current_cost < loss:
                    ok = True
                else:
                    t *= beta

                if t < 1e-20:
                    # Stopping criterion
                    self.prior.data = prior_copy
                    if state:
                        self.steps.data = sigmas_copy
                    ok = True
                    end = True

        # Avoiding numerical instabitility in the step size
        future_step = min(10*t, 1e4)
        return future_step, end


class DeepPrimalDualPriorLearningConv1d(DeepPrimalDualPriorLearning):
    """
    Analysis prior learning with 1D convolutions.
    Inherits from DeepPrimalDualPriorLearning.

    Parameters
    ----------
    kernel_size : int
        Size of the convolutional kernel.
        Default : 3.

    Attributes
    ----------
    kernel_size : int
        Size of the convolutional kernel.
    conv : function
        1D convolution with pytorch
    convt : function
        1D correlation with pytorch
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 kernel_size=3, device=None, learn_steps=True):
        super().__init__(n_components, n_iter, lambd, None,
                         device, learn_steps)

        # Conv function
        self.conv = torch.nn.functional.conv1d
        self.convt = torch.nn.functional.conv_transpose1d

        self.constraint = "conv"
        self.kernel_size = kernel_size

    def compute_lipschitz(self):
        """ Computes the Lipschitz constant using the FFT """
        with torch.no_grad():
            fourier_prior = fft.fft(self.prior)
            self.lipschitz = torch.max(
                torch.real(fourier_prior * torch.conj(fourier_prior))
                )

    def rescale(self):
        """ Rescales conv kernels """
        with torch.no_grad():
            norm = torch.norm(self.prior)
            self.prior /= norm
        return norm

    def forward(self, y):
        """
        Condat-Vu-like forward pass with convolutions

        Parameters
        ----------
        y : torch.Tensor, shape (dim_y, number of data)
            Data to be processed by (F)ISTA

        Returns
        -------
        out_p : torch.Tensor, shape (dim_x, number of data)
            Approximation of the signal associated to y
        """

        out_p = torch.zeros_like(y)

        out_d = torch.zeros(
            (y.shape[1], self.n_components, y.shape[0] - self.kernel_size + 1),
            dtype=torch.float,
            device=self.device
            )

        self.taus = 1 / (0.5 * self.norm_A + self.steps * self.lipschitz)

        for i in range(self.n_iter-1):
            # Keep last primal iterate
            out_old = out_p.clone()

            # Gradient descent primal
            first_gradient = torch.matmul(
                self.operator.t(), torch.matmul(self.operator, out_p) - y
                )
            second_gradient = self.convt(out_d, self.prior)
            gradient = first_gradient + second_gradient.sum(axis=1).t()
            out_p = out_p - self.taus[i] * gradient

            # Gradient ascent dual
            ascent = out_d + self.steps[i] * self.conv(
                (2 * out_p - out_old).t()[:, None, :], self.prior
                )
            out_d = ascent / torch.max(torch.ones_like(ascent),
                                       torch.abs(ascent) / self.lambd)

        first_gradient = torch.matmul(
            self.operator.t(), torch.matmul(self.operator, out_p) - y
            )
        second_gradient = self.convt(out_d, self.prior)
        gradient = first_gradient + second_gradient.sum(axis=1).t()
        out_p = out_p - self.taus[-1] * gradient

        return out_p

    def cost(self, y, x):
        """ LASSO cost function """
        z = self.conv(x.t()[:, None, :], self.prior)
        return lasso(y, x, z, self.operator, self.lambd)


##############
# Inpainting #
##############


class DPDPLInpainting(DeepPrimalDualPriorLearning):
    """
    Analysis prior learning, based on Condat-Vu algorithm, for inpainting.
    Inherits from DeepPrimalDualPriorLearning.
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 D=None, device=None, learn_steps=True):
        super().__init__(n_components, n_iter, lambd, D,
                         device, learn_steps)

    def forward(self, y):
        """
        Condat-Vu-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (dim_y, number of data)
            Data to be processed by (F)ISTA

        Returns
        -------
        out_p : torch.Tensor, shape (dim_x, number of data)
            Approximation of the signal associated to y
        """
        out_p = torch.zeros_like(y)

        out_d = torch.zeros((self.prior.shape[0], y.shape[1]),
                            dtype=torch.float,
                            device=self.device)

        self.taus = 1 / (0.5 + self.steps * self.lipschitz)

        for i in range(self.n_iter-1):

            out_p, out_d = condat_vu_iteration(
                out_p,
                out_d,
                y,
                self.operator,
                self.prior,
                self.steps[i],
                self.taus[i],
                self.lambd,
                True
            )

        # Last iteration
        # Only out_p is updated, the full iteration is not called
        # to prevent issues linked to null gradients.
        first_term = self.operator * (out_p - y)
        second_term = torch.matmul(self.prior.t(), out_d)
        out_p = out_p - self.taus[-1] * (first_term + second_term)

        return out_p

    def cost(self, y, x):
        """ LASSO cost function """
        z = torch.matmul(self.prior, x)
        return lasso(y, x, z, self.operator, self.lambd, inpainting=True)

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


class DPDPLInpaintingUNTF(DeepPrimalDualPriorLearningUNTF):
    """
    Analysis prior learning with UNTF constraint,
    based on Condat-Vu algorithm, for inpainting.
    Inherits from DeepPrimalDualPriorLearningUNTF.
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 init_D=None, device=None, learn_steps=True):
        super().__init__(n_components, n_iter, lambd, init_D,
                         device, learn_steps)

    def forward(self, y):
        """
        Condat-Vu-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (dim_y, number of data)
            Data to be processed by (F)ISTA

        Returns
        -------
        out_p : torch.Tensor, shape (dim_x, number of data)
            Approximation of the signal associated to y
        """
        out_p = torch.zeros_like(y)

        out_d = torch.zeros((self.prior.shape[0], y.shape[1]),
                            dtype=torch.float,
                            device=self.device)

        self.taus = 1 / (0.5 + self.steps * self.lipschitz)

        for i in range(self.n_iter-1):

            out_p, out_d = condat_vu_iteration(
                out_p,
                out_d,
                y,
                self.operator,
                self.prior,
                self.steps[i],
                self.taus[i],
                self.lambd,
                True
            )

        # Last iteration
        # Only out_p is updated, the full iteration is not called
        # to prevent issues linked to null gradients.
        first_term = self.operator * (out_p - y)
        second_term = torch.matmul(self.prior.t(), out_d)
        out_p = out_p - self.taus[-1] * (first_term + second_term)

        return out_p

    def cost(self, y, x):
        """
        LASSO cost function
        """
        z = torch.matmul(self.prior, x)
        return lasso(y, x, z, self.operator, self.lambd, inpainting=True)

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


class DPDPLInpaintingConv(DeepPrimalDualPriorLearningConv1d):
    """
    Analysis prior learning with convolutions,
    based on Condat-Vu algorithm, for inpainting.
    Inherits from DeepPrimalDualPriorLearningConv1d.
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 kernel_size=3, device=None, learn_steps=True):
        super().__init__(n_components, n_iter, lambd, kernel_size,
                         device, learn_steps)

        # Conv function
        self.conv = torch.nn.functional.conv2d
        self.convt = torch.nn.functional.conv_transpose2d

    def rescale(self):
        """ Rescales conv kernels """
        with torch.no_grad():
            if self.conv_constraint:
                self.prior[:, :, 0, 0] = 0
                self.prior[:, :, -1, 0] = 0
                self.prior[:, :, 0, -1] = 0
                self.prior[:, :, -1, -1] = 0
            norm = torch.norm(self.prior)
            self.prior /= norm
        return norm

    def forward(self, y):
        """
        Condat-Vu-like forward pass

        Parameters
        ----------
        y : torch.Tensor, shape (dim_y, number of data)
            Data to be processed by (F)ISTA

        Returns
        -------
        out_p : torch.Tensor, shape (dim_x, number of data)
            Approximation of the signal associated to y
        """
        out_p = torch.zeros_like(y)

        out_d = torch.zeros(
            (y.shape[0], self.n_components,
             y.shape[1] - self.kernel_size + 1,
             y.shape[2] - self.kernel_size + 1),
            dtype=torch.float,
            device=self.device
            )

        self.taus = 1 / (0.5 + self.steps * self.lipschitz)

        for i in range(self.n_iter-1):
            # Keep last primal iterate
            out_old = out_p.clone()

            # Gradient descent primal
            first_gradient = self.operator * (out_p - y)
            second_gradient = self.convt(out_d, self.prior)
            gradient = first_gradient + second_gradient.sum(axis=1)
            out_p = out_p - self.taus[i] * gradient

            # Gradient ascent dual
            ascent = out_d + self.steps[i] * self.conv(
                (2 * out_p - out_old)[:, None, :, :], self.prior
                )
            out_d = ascent / torch.max(torch.ones_like(ascent),
                                       torch.abs(ascent) / self.lambd)

        first_gradient = self.operator * (out_p - y)
        gradient = first_gradient + second_gradient.sum(axis=1)
        out_p = out_p - self.taus[-1] * gradient

        return out_p

    def cost(self, y, x):
        """
        LASSO cost function
        """
        z = self.conv(x[:, None, :, :], self.prior)
        return lasso(y, x, z, self.operator, self.lambd, inpainting=True)

    def fit(self, data_y, A, conv_constraint=False):
        """
        Training procedure
        """

        # Dimension
        self.dim_y = data_y.shape[0]
        self.dim_x = self.dim_y

        # Operator
        self.operator = torch.from_numpy(A).float().to(self.device)

        # Prior
        self.prior = nn.Parameter(
            torch.rand(
                (self.n_components, 1, self.kernel_size, self.kernel_size),
                dtype=torch.float,
                device=self.device
                )
        )

        # Constraint
        self.conv_constraint = conv_constraint

        # Scaling and computing step
        self.rescale()
        self.steps = nn.Parameter(
            torch.ones(self.n_iter, device=self.device, dtype=torch.float)
            )
        self.taus = torch.zeros_like(self.steps)
        self.compute_lipschitz()

        # Data
        self.Y_tensor = torch.from_numpy(data_y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss


#####################
# Separable filters #
#####################


class DPDPLInpaintingConvSeparable(DPDPLInpaintingConv):
    """
    Analysis prior learning with separable convolutions,
    based on Condat-Vu algorithm, for inpainting.
    Inherits from DPDPLInpaintingConv.
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 kernel_size=3, device=None, learn_steps=True):
        super().__init__(n_components, n_iter, lambd, kernel_size,
                         device, learn_steps)
        self.n_components = 2

    def rescale(self):
        """ Rescales conv kernels """
        with torch.no_grad():
            self.prior[0, :, :, 0] = 0
            self.prior[0, :, :, -1] = 0
            self.prior[1, :, 0, :] = 0
            self.prior[1, :, -1, :] = 0
            norm = torch.norm(self.prior)
            self.prior /= norm
        return norm
