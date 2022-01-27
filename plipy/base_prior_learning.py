import numpy as np
import torch
import torch.nn as nn
import time


class BasePriorLearning(nn.Module):
    """
    Base class for prior learning algorithms

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
    """
    def __init__(self, n_components, n_iter, lambd=0.1,
                 init_D=None, device=None, learn_steps=False):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Regularization parameter
        self.lambd = lambd

        # Algorithm parameters
        self.n_iter = n_iter
        self.learn_steps = learn_steps

        # Number of atoms
        self.n_components = n_components

        # Shape
        self.dim_x = n_components
        self.dim_y = None
        self.dim_signal = None

        # Initial prior
        self.init_D = init_D
        self.lipschitz = 0.

        # Tensors
        self.operator = None
        self.prior = None
        self.steps = None
        self.Y_tensor = None

        # Parameters for experiments
        self.keep_dico = False

    def get_prior(self):
        """ Returns the current prior """
        return self.prior.to("cpu").detach().numpy()

    def rescale(self, atoms="columns"):
        """
        Constrains the prior to have normalized atoms

        Returns
        -------
        norm_atoms : torch.Tensor, shape (dim_x)
            Contains the norms of the current atoms.
        """
        with torch.no_grad():
            if atoms == "columns":
                norm_atoms = torch.norm(self.prior, dim=0)
            elif atoms == "rows":
                norm_atoms = torch.norm(self.prior, dim=1)[:, None]
            norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1
            self.prior /= norm_atoms
        return norm_atoms

    def unscale(self, norm_atoms):
        """
        Cancels the scaling using norms previously computed

        Parameters
        ----------
        norm_atoms : tocrh.Tensor, shape (dim_x)
            Contains the norms of the current atoms.
            Computed by rescale()
        """
        with torch.no_grad():
            self.prior *= norm_atoms

    def compute_lipschitz(self):
        """ Computes an upper bound of the Lipschitz constant of the prior"""
        with torch.no_grad():
            self.lipschitz = torch.norm(
                torch.matmul(self.prior.t(), self.prior)
                ).item()
            if self.lipschitz == 0:
                self.lipschitz = 1.

    def forward(self, y):
        """
        Main algorithm.

        Parameters
        ----------
        y : torch.Tensor, shape (dim_y, number of data)
            Data to be processed
        """
        raise NotImplementedError

    def cost(self, y, x):
        """ Cost function """
        raise NotImplementedError

    def line_search(self, step, loss, state):
        """
        Gradient descent step with line search

        Parameters
        ----------
        step : float
            Starting step for line search.
        loss : float
            Current value of the loss.
        state : int
            Indicates if the steps sizes have to be optimized.
            0 : only the prior is optimized.
            1 : both the prior and the step sizes are optimized.

        Returns
        -------
        future_step : float
            Value of the future starting step size.
        end : bool
            True if the optimization is done, False otherwise.
        """
        # Line search parameters
        beta = 0.5

        # Learning rate
        t = step

        ok = False
        end = False
        norm_atoms = None

        with torch.no_grad():
            # Learning step
            self.prior -= beta * t * self.prior.grad

            if state:
                self.steps -= beta * t * self.steps.grad

            init = True

            while not ok:
                if not init:
                    # Unscaling
                    self.unscale(norm_atoms)
                    # Backtracking
                    self.prior -= (beta-1) * t * self.prior.grad
                    if state:
                        self.steps -= (beta-1) * t * self.steps.grad
                else:
                    init = False

                # Rescaling
                norm_atoms = self.rescale()

                # Computing step
                self.compute_lipschitz()

                # Computing loss with new parameters
                current_cost = self.cost(self.Y_tensor,
                                         self.forward(self.Y_tensor)).item()

                if current_cost < loss:
                    ok = True
                else:
                    t *= beta

                if t < 1e-20:
                    # Unscaling
                    self.unscale(norm_atoms)
                    # Stopping criterion
                    self.prior += t * self.prior.grad
                    if state:
                        self.steps += t * self.steps.grad
                    # Rescaling
                    self.rescale()

                    # Computing step
                    self.compute_lipschitz()

                    ok = True
                    end = True

        # Avoiding numerical instabitility in the step size
        future_step = min(10*t, 1e4)
        return future_step, end

    def training_process(self, backprop=True, tol=1e-6):
        """
        Training function, with backtracking line search.

        Returns
        -------
        loss.item() : float
            Final value of the loss after training.
        """
        # Initial backtracking step
        step = 1
        end = False
        state = 0

        old_loss = None
        self.path_optim = []
        self.path_loss = []
        self.path_times = [0]
        start = time.time()

        while not end:
            # Keep track of the dictionaries
            if self.keep_dico:
                self.path_optim.append(self.get_prior())
            # Computing loss and gradients
            if backprop:
                out = self.forward(self.Y_tensor)
            else:
                with torch.no_grad():
                    out = self.forward(self.Y_tensor)
            loss = self.cost(self.Y_tensor, out)
            if self.keep_dico:
                self.path_loss.append(loss.item())
            loss.backward()

            # Line search
            step, end = self.line_search(step, loss, state)

            # Checking termination
            if old_loss is not None:
                if np.abs(old_loss - loss.item()) / old_loss < tol:
                    end = True

            if end and not state and self.learn_steps:
                state = 1
                end = False
                step = 1

            old_loss = loss.item()

            # Putting the gradients to zero
            self.prior.grad.zero_()

            if state:
                self.steps.grad.zero_()

            if self.keep_dico:
                self.path_times.append(time.time() - start)

        if self.keep_dico:
            self.path_optim.append(self.get_prior())

        return loss.item()

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

        # Data
        self.Y_tensor = torch.from_numpy(data_y).float().to(self.device)

        # Training
        loss = self.training_process()
        return loss

    def eval(self):
        """ Computes the results with one forward pass from the data """
        with torch.no_grad():
            return self.forward(self.Y_tensor).to("cpu").detach().numpy()
