from sklearn.decomposition import DictionaryLearning


class DL():
    """
    Dictionary Learning.
    Adapted from sklearn.decomposition.DictionaryLearning.

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

    Attributes
    ----------
    D : np.array, shape (dim_signal, dim_x)
        Current dictionary
    """
    def __init__(self, n_components, n_iter=None, lambd=0.1, init_D=None):
        self.algo = DictionaryLearning(n_components=n_components,
                                       alpha=lambd,
                                       fit_algorithm="cd",
                                       transform_algorithm="lasso_cd",
                                       dict_init=init_D.T,
                                       transform_alpha=lambd)

        self.D = init_D

    def get_prior(self):
        return self.D

    def fit(self, data_y):
        self.code = self.algo.fit_transform(data_y.T).T
        self.D = self.algo.components_.T

    def eval(self):
        return self.code
