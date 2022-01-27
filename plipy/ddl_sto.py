import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import time

from .ddl import DeepDictionaryLearning, DeepCDL
from alphacsc.init_dict import init_dictionary


class SignalDataset(torch.utils.data.Dataset):
    """
    Dataset for stochastic DDL

    Parameters
    ----------
    data : np.array
        Data to be processed

    Attributes
    ----------
    data : np.array
        Data to be processed
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[:, idx].reshape(-1, 1)

    def __len__(self):
        return self.data.shape[1]


class StoDeepDictionaryLearning(DeepDictionaryLearning):
    """
    Stochastic version of Deep Dictionary Learning.

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

    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 init_D=None, device=None, learn_steps=True,
                 algo="fista"):
        super().__init__(n_components, n_iter, lambd, init_D,
                         device, learn_steps, lambd)

    def forward(self, y):
        """
        ISTA-like forward pass
        """
        out_shape = (y.shape[0], self.dim_x, y.shape[2])
        out = torch.zeros(out_shape,
                          dtype=torch.float,
                          device=self.device)

        if self.algo == "fista":
            out_old = out.clone()
            t_old = 1.

        steps = self.steps / self.lipschitz
        product = torch.matmul(self.operator, self.prior)

        for i in range(self.n_iter):
            # Gradient descent
            out = out - steps[i] * torch.matmul(product.t(),
                                                torch.matmul(product,
                                                out) - y)
            # Thresholding
            thresh = torch.abs(out) - steps[i] * self.lambd
            out = torch.sign(out) * F.relu(thresh)

            if self.algo == "fista":
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
                z = out + ((t_old-1) / t) * (out - out_old)
                out_old = out.clone()
                t_old = t
                out = z

        return out

    def stoch_line_search(self, batch, eta, loss, state):
        """
        Stochastic line search gradient descent
        """
        ok = False
        norm_col = None
        old_eta = eta

        if not state:
            norm_grad = torch.sum(self.prior.grad ** 2)
        elif state:
            norm_grad = torch.sum(self.prior.grad ** 2)\
                + torch.sum(self.steps.grad ** 2)

        with torch.no_grad():
            # Learning step
            self.prior -= self.beta * eta * self.prior.grad

            if state:
                self.steps -= self.beta * eta * self.steps.grad

            init = True

            while not ok:
                if not init:
                    # Unscaling
                    self.unscale(norm_col)
                    # Backtracking
                    self.prior -= (self.beta-1)\
                        * eta * self.prior.grad
                    if state:
                        self.steps -= (self.beta-1) * eta * self.steps.grad
                else:
                    init = False

                # Rescaling
                norm_col = self.rescale()

                # Computing step
                self.compute_lipschitz()

                # Computing loss with new parameters
                current_cost = self.cost(batch, self.forward(batch)).item()

                if current_cost < loss - self.c * eta * norm_grad:
                    ok = True
                else:
                    eta *= self.beta

                if eta < 1e-20:
                    # Stopping criterion
                    self.prior += eta * self.prior.grad
                    if state:
                        self.steps += eta * self.steps.grad
                    ok = True

        return old_eta

    def train(self, epochs, state):
        """
        Training function, with backtracking line search
        """

        for i in range(epochs):
            avg_loss = 0

            for idx, data in enumerate(self.dataloader):

                if self.iterations_per_epoch is not None:
                    if idx >= self.iterations_per_epoch:
                        break

                if self.keep_dico and not (i == 0 and state and idx == 0):
                    self.path_optim.append(self.get_prior())

                if self.device != "cpu":
                    data = data.cuda(self.device)

                data = data.float()

                # Computing loss and gradients
                out = self.forward(data)
                loss = self.cost(data, out)
                if self.keep_dico:
                    self.path_loss.append(loss.item())
                loss.backward()

                avg_loss = idx * avg_loss / (idx+1)\
                    + (1 / (idx+1)) * loss.item()

                # Optimizing
                if i == 0:
                    eta = self.etamax
                else:
                    eta *= self.gamma **\
                        (self.mini_batch_size / self.batch_size)

                eta = self.stoch_line_search(data, eta, loss.item(), state)

                # Putting the gradients to zero
                self.prior.grad.zero_()
                if state:
                    self.steps.grad.zero_()

                if self.keep_dico:
                    self.path_times.append(time.time() - self.start)

                if self.time_limit is not None:
                    if time.time() - self.start > self.time_limit:
                        if self.keep_dico:
                            self.path_optim.append(self.get_prior())
                        return loss.item()

        if self.keep_dico:
            self.path_optim.append(self.get_prior())

        return loss.item()

    def fit(self, data_y, A=None, epochs=10, iterations_per_epoch=None,
            mini_batch_size=1000, etamax=1, c=None, beta=0.5,
            gamma=0.5, epochs_step_size=10, time_limit=None):
        """
        Training procedure
        """
        # Dimension
        self.dim_y = data_y.shape[0]

        if A is None:
            self.dim_signal = self.dim_y
        else:
            self.dim_signal = A.shape[1]

        # Operator
        if A is None:
            self.operator = torch.eye(self.dim_y,
                                      device=self.device,
                                      dtype=torch.float)
        else:
            self.operator = torch.from_numpy(A).float().to(self.device)

        # Dictionary
        if self.init_D is None:
            choice = np.random.choice(data_y.shape[1], self.n_components)
            dico = data_y[:, choice]
            self.prior = nn.Parameter(
                torch.tensor(dico, device=self.device, dtype=torch.float)
            )
        else:
            dico_tensor = torch.from_numpy(self.init_D).float().to(self.device)
            self.prior = nn.Parameter(dico_tensor)

        # Scaling and computing step
        self.rescale()
        self.steps = nn.Parameter(
            torch.ones(self.n_iter, device=self.device, dtype=torch.float)
        )
        self.compute_lipschitz()

        if mini_batch_size is None or mini_batch_size > data_y.shape[1]:
            self.mini_batch_size = data_y.shape[1]
        else:
            self.mini_batch_size = mini_batch_size

        self.iterations_per_epoch = iterations_per_epoch

        if self.iterations_per_epoch is None:
            self.batch_size = data_y.shape[1]
        else:
            self.batch_size = self.mini_batch_size * self.iterations_per_epoch

        if c is None:
            # Heuristic
            self.c = 10 / self.mini_batch_size
        else:
            self.c = c

        self.time_limit = time_limit

        self.etamax = etamax
        self.data_size = data_y.shape[1]
        self.beta = beta
        self.gamma = gamma

        # Dataset
        dataset = SignalDataset(data_y)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            pin_memory=True
        )

        if self.keep_dico:
            self.path_optim = []
            self.path_times = [0]
            self.path_loss = []
            self.start = time.time()

        # Learning dictionary
        loss = self.train(epochs, state=0)

        # Learning step sizes
        if self.learn_steps:
            loss = self.train(epochs=epochs_step_size, state=1)

        return loss

    def eval(self, data_y, batch_size=10000):
        """
        Compute results
        """
        dataset_eval = SignalDataset(data_y)
        self.dataloader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        results = []
        with torch.no_grad():
            for _, data in enumerate(self.dataloader_eval):
                if self.device != "cpu":
                    data = data.cuda(self.device)
                data = data.float()
                results.append(
                    self.forward(data).to("cpu").detach().numpy()[:, :, 0].T
                )
        return np.concatenate(results, axis=1)


class ConvSignalDataset(torch.utils.data.Dataset):
    """
    Dataset for stochastic DCDL

    Parameters
    ----------
    data : np.array
        Data to be processed

    Attributes
    ----------
    data : np.array
        Data to be processed
    """
    def __init__(self, data, window):
        super().__init__()
        self.data = data
        self.window = window

    def __getitem__(self, idx):
        return self.data[:, idx:(idx+self.window)]

    def __len__(self):
        return self.data.shape[1] - self.window


class StoDeepCDL(DeepCDL, StoDeepDictionaryLearning):
    def __init__(self, n_components, n_iter, lambd=0.1, kernel_size=5,
                 device=None, learn_steps=True, algo="fista"):
        super().__init__(n_components, n_iter, lambd, kernel_size,
                         device, learn_steps, algo)

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
             y.shape[2] - self.kernel_size + 1),
            dtype=torch.float,
            device=self.device
        )

        if self.algo == "fista":
            out_old = out.clone()
            t_old = 1

        steps = self.steps / self.lipschitz

        D = self.prior * self.window_tukey

        for i in range(self.n_iter):
            # Gradient descent
            result1 = self.convt(out, D)
            result2 = self.conv(
                (result1 - y),
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
        D = self.prior * self.window_tukey

        signal = self.convt(x, D)
        res = signal - y
        l2 = (res * res).sum()
        l1 = torch.abs(x).sum()

        return 0.5 * l2 + self.lambd * l1

    def train(self, epochs, state):
        return StoDeepDictionaryLearning.train(self, epochs, state)

    def fit(self, data_y, window, epochs=10, iterations_per_epoch=None,
            mini_batch_size=1000, etamax=1, c=None, beta=0.5,
            gamma=0.5, epochs_step_size=10):
        """
        Training procedure
        """
        # Dimension
        self.dim_y = data_y.shape[1]
        self.n_channels = data_y.shape[0]

        if window is None:
            self.window = 10 * self.kernel_size
        else:
            self.window = window

        data_y_norm = data_y / data_y.std()

        dico_init = init_dictionary(
            data_y_norm[None, :, :],
            self.n_components,
            self.kernel_size,
            rank1=False
        )

        # Dictionary
        self.prior = nn.Parameter(
            torch.tensor(
                dico_init,
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

        # Optimization parameters
        if mini_batch_size is None or mini_batch_size > data_y.shape[1]:
            self.mini_batch_size = data_y.shape[1] - self.window
        else:
            self.mini_batch_size = mini_batch_size

        self.iterations_per_epoch = iterations_per_epoch
        self.batch_size = self.mini_batch_size * self.iterations_per_epoch

        if c is None:
            # Heuristic
            self.c = 10 / self.mini_batch_size
        else:
            self.c = c

        self.etamax = etamax
        # self.data_size = data_y.shape[1]
        self.beta = beta
        self.gamma = gamma

        # Dataset
        dataset = ConvSignalDataset(data_y_norm, self.window)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            pin_memory=True
        )

        with torch.no_grad():
            sample = next(iter(self.dataloader)).cuda(self.device).float()
            self.lambd *= torch.max(torch.abs(self.conv(sample, self.prior)))

        # Learning dictionary
        loss = self.train(epochs, state=0)

        # Learning step sizes
        if self.learn_steps:
            loss = self.train(epochs=epochs_step_size, state=1)

        return loss


class StoDeepCDL1Rank(StoDeepCDL):
    def __init__(self, n_components, n_iter, lambd=0.1, kernel_size=5,
                 device=None, learn_steps=True, algo="fista"):
        super().__init__(n_components, n_iter, lambd, kernel_size,
                         device, learn_steps, algo)

    def rescale(self):
        """
        Constrains the dictionary to have normalized atoms
        """
        with torch.no_grad():
            norm_col_u = torch.norm(self.u, dim=1, keepdim=True)
            norm_col_u[torch.nonzero((norm_col_u == 0), as_tuple=False)] = 1
            self.u /= norm_col_u

            norm_col_v = torch.norm(self.v, dim=2, keepdim=True)
            norm_col_v[torch.nonzero((norm_col_v == 0), as_tuple=False)] = 1
            self.v /= norm_col_v
        return norm_col_v, norm_col_u

    def unscale(self, norm_v, norm_u):
        """
        Cancels the scaling using norms previously computed
        """
        with torch.no_grad():
            self.v *= norm_v
            self.u *= norm_u

    def get_prior(self):
        with torch.no_grad():
            D = self.v * self.window_tukey
        return self.u.to("cpu").detach().numpy(), D.to("cpu").detach().numpy()

    def compute_lipschitz(self):
        """ Compute the Lipschitz constant using the FFT"""
        with torch.no_grad():
            self.prior = self.u * self.v
            fourier_prior = fft.fft(self.prior, dim=2)
            self.lipschitz = torch.max(
                    torch.real(fourier_prior * torch.conj(fourier_prior)),
                    dim=2
                    )[0].sum().item()
            if self.lipschitz == 0:
                self.lipschitz = 1

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
             y.shape[2] - self.kernel_size + 1),
            dtype=torch.float,
            device=self.device
        )

        if self.algo == "fista":
            out_old = out.clone()
            t_old = 1

        steps = self.steps / self.lipschitz
        self.prior = self.u * self.v

        D = self.prior * self.window_tukey

        for i in range(self.n_iter):
            # Gradient descent
            result1 = self.convt(out, D)
            result2 = self.conv(
                (result1 - y),
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

    def stoch_line_search(self, batch, eta, loss, state):
        """
        Stochastic line search gradient descent
        """
        ok = False
        norm_u = None
        norm_v = None
        old_eta = eta

        if not state:
            norm_grad = torch.sum(self.v.grad ** 2)\
                + torch.sum(self.u.grad ** 2)
        elif state:
            norm_grad = torch.sum(self.v.grad ** 2)\
                + torch.sum(self.u.grad ** 2)\
                + torch.sum(self.steps.grad ** 2)

        with torch.no_grad():
            # Learning step
            self.v -= self.beta * eta * self.v.grad
            self.u -= self.beta * eta * self.u.grad

            if state:
                self.steps -= self.beta * eta * self.steps.grad

            init = True

            while not ok:
                if not init:
                    # Unscaling
                    self.unscale(norm_v, norm_u)
                    # Backtracking
                    self.v -= (self.beta-1)\
                        * eta * self.v.grad
                    self.u -= (self.beta-1)\
                        * eta * self.u.grad
                    if state:
                        self.steps -= (self.beta-1) * eta * self.steps.grad
                else:
                    init = False

                # Rescaling
                norm_v, norm_u = self.rescale()

                # Computing step
                self.compute_lipschitz()

                # Computing loss with new parameters
                current_cost = self.cost(batch, self.forward(batch)).item()

                if current_cost < loss - self.c * eta * norm_grad:
                    ok = True
                else:
                    eta *= self.beta

                if eta < 1e-20:
                    # Stopping criterion
                    self.v += eta * self.v.grad
                    self.u += eta * self.u.grad
                    if state:
                        self.steps += eta * self.steps.grad
                    ok = True

        return old_eta

    def train(self, epochs, state):
        """
        Training function, with backtracking line search
        """

        for i in range(epochs):
            avg_loss = 0

            for idx, data in enumerate(self.dataloader):

                if self.iterations_per_epoch is not None:
                    if idx >= self.iterations_per_epoch:
                        break

                if self.keep_dico and not (i == 0 and state and idx == 0):
                    self.path_optim.append(self.get_prior())

                if self.device != "cpu":
                    data = data.cuda(self.device)

                data = data.float()

                # Computing loss and gradients
                out = self.forward(data)
                loss = self.cost(data, out)
                if self.keep_dico:
                    self.path_loss.append(loss.item())
                loss.backward()

                avg_loss = idx * avg_loss / (idx+1)\
                    + (1 / (idx+1)) * loss.item()

                # Optimizing
                if i == 0:
                    eta = self.etamax
                else:
                    eta *= self.gamma **\
                        (self.mini_batch_size / self.batch_size)

                eta = self.stoch_line_search(data, eta, loss.item(), state)

                # Putting the gradients to zero
                self.v.grad.zero_()
                self.u.grad.zero_()
                if state:
                    self.steps.grad.zero_()

                if self.keep_dico:
                    self.path_times.append(time.time() - self.start)

            print(avg_loss)

        if self.keep_dico:
            self.path_optim.append(self.get_prior())

        return loss.item()

    def fit(self, data_y, window, epochs=10, iterations_per_epoch=None,
            mini_batch_size=1000, etamax=1, c=None, beta=0.5,
            gamma=0.5, epochs_step_size=10):
        """
        Training procedure
        """
        # Dimension
        self.dim_y = data_y.shape[1]
        self.n_channels = data_y.shape[0]

        if window is None:
            self.window = 10 * self.kernel_size
        else:
            self.window = window

        data_y_norm = data_y / data_y.std()

        init = init_dictionary(
            data_y_norm[None, :, :], self.n_components, self.kernel_size
        )

        u = init[:, :self.n_channels][:, :, None]
        v = init[:, self.n_channels:][:, None, :]

        self.u = nn.Parameter(
            torch.tensor(
                u,
                dtype=torch.float,
                device=self.device
            )
        )

        self.v = nn.Parameter(
            torch.tensor(
                v,
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

        # Optimization parameters
        if mini_batch_size is None or mini_batch_size > data_y.shape[1]:
            self.mini_batch_size = data_y.shape[1] - self.window
        else:
            self.mini_batch_size = mini_batch_size

        self.iterations_per_epoch = iterations_per_epoch
        self.batch_size = self.mini_batch_size * self.iterations_per_epoch

        if c is None:
            # Heuristic
            self.c = 10 / self.mini_batch_size
        else:
            self.c = c

        self.etamax = etamax
        # self.data_size = data_y.shape[1]
        self.beta = beta
        self.gamma = gamma

        # Dataset
        dataset = ConvSignalDataset(data_y_norm, self.window)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            pin_memory=True
        )

        with torch.no_grad():
            sample = next(iter(self.dataloader)).cuda(self.device).float()
            self.lambd *= torch.max(
                torch.abs(self.conv(sample, self.u * self.v))
            )

        # Learning dictionary
        loss = self.train(epochs, state=0)

        # Learning step sizes
        if self.learn_steps:
            loss = self.train(epochs=epochs_step_size, state=1)

        return loss
