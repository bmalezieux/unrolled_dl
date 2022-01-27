import numpy as np

from . import create_patches_overlap, patch_average
from .ddl import DDLInpainting, DDLInpaintingConv
from .dpdpl import (DPDPLInpainting,
                    DPDPLInpaintingUNTF,
                    DPDPLInpaintingConv)


class APLInpainting():
    """
    Inpainting using Analysis prior learning

    Parameters
    ----------
    y : np.array
        Image
    A : np.array
        Mask
    m : int
        Dimension of the patch
    n_components : int
        Number of atoms in the prior.
    n_iter : int
        Number of unrolled iterations.
    lambd : float
        Regularization parameter.
        Default : 0.1.
    constraint : str
        Constraint for Analysis ["untf", "un"]

    Attributes
    ----------
    m : int
        Dimension of the patch
    r, c : int, int
        Dimensions of the image
    y : np.array
        Image
    A : np.array
        Mask
    m : int
        Dimension of the patch
    n_components : int
        Number of atoms in the prior.
    n_iter : int
        Number of unrolled iterations.
    constraint : str
        Constraint for Analysis ["untf", "un"]
    D : np.array
        Current dictionary / prior

    """
    def __init__(self, y, n_iter=20, A=None, lambd=0.1, m=10,
                 n_components=None, constraint="untf"):
        self.m = m
        self.r, self.c = y.shape
        self.y, self.A = create_patches_overlap(y, m, A)
        self.lambd = lambd
        self.n_iter = n_iter
        self.D = None
        if n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = m * m
        self.constraint = constraint

    def get_prior(self):
        return self.D

    def fit(self):
        choice = np.random.choice(self.y.shape[1], self.n_components)
        D0 = self.y[:, choice].T
        if self.constraint == "untf":
            dpdpl = DPDPLInpaintingUNTF(self.n_components,
                                        self.n_iter,
                                        lambd=self.lambd,
                                        init_D=D0)
        elif self.constraint == "un":
            dpdpl = DPDPLInpainting(self.n_components,
                                    self.n_iter,
                                    lambd=self.lambd,
                                    init_D=D0)
        loss = dpdpl.fit(self.y, self.A)
        self.D = dpdpl.get_prior()
        self.result = dpdpl.eval()
        im_result = patch_average(self.result, self.m, self.r, self.c)
        return im_result, loss


class APLInpaintingConv():
    """
    Inpainting using convolutional Analysis prior learning
    """
    def __init__(self, y, n_iter=20, A=None, lambd=0.1, m=4,
                 n_components=50):
        self.m = m
        self.y = y
        self.A = A
        self.lambd = lambd
        self.n_iter = n_iter
        self.D = None
        if n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = m * m

    def fit(self):
        dpdpl_inpainting_image = DPDPLInpaintingConv(self.n_components,
                                                     self.n_iter,
                                                     lambd=self.lambd,
                                                     kernel_size=self.m)
        loss = dpdpl_inpainting_image.fit(self.y[None, :, :],
                                          self.A[None, :, :])
        im_result_conv = np.clip(dpdpl_inpainting_image.eval(), 0, 1)[0]
        return im_result_conv, loss


class SPLInpainting():
    """
    Inpainting using synthesis prior learning
    """
    def __init__(self, y, n_iter=20, A=None, lambd=1, m=12,
                 n_components=None, algo="fista"):
        self.m = m
        self.r, self.c = y.shape
        self.y, self.A = create_patches_overlap(y, m, A)
        self.lambd = lambd
        self.n_iter = n_iter
        self.D = None

        if n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = m * m

        self.algo = algo

    def get_prior(self):
        return self.D

    def fit(self):
        ddl = DDLInpainting(self.n_components,
                            self.n_iter,
                            lambd=self.lambd)
        loss = ddl.fit(self.y, self.A)
        self.D = ddl.get_prior()
        result = ddl.eval()
        im_result = patch_average(self.D @ result, self.m, self.r, self.c)
        return im_result, loss


class SPLInpaintingConv():
    """
    Inpainting using convolutional synthesis prior learning
    """
    def __init__(self, y, n_iter=20, A=None, lambd=0.1,
                 m=10, n_components=None):
        self.m = m
        self.y = y
        self.A = A
        self.lambd = lambd
        self.n_iter = n_iter
        self.D = None
        if n_components is not None:
            self.n_components = n_components
        else:
            self.n_components = m * m

    def fit(self):
        ddl_inpainting = DDLInpaintingConv(self.n_components,
                                           self.n_iter,
                                           lambd=self.lambd,
                                           kernel_size=self.m)
        loss = ddl_inpainting.fit(self.y[None, :, :],
                                  self.A[None, :, :])
        im_result_conv = np.clip(ddl_inpainting.eval(), 0, 1)[0]
        return im_result_conv, loss
