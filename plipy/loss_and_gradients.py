import numpy as np
import torch
import torch.nn.functional as F


def ista_iteration(iterate, y, operator, step, lambd,
                   inpainting=False, prior_inpainting=None):
    """
    ISTA iteration

    Parameters
    ----------
    iterate : torch.Tensor
        Current iterate
    y : torch.Tensor
        data
    operator : torch.Tensor
        operator
    step : torch.Tensor, shape (1)
        Step size
    lambd : float
        Regularization parameter
    inpainting : bool
        If set to True, the operator is
        considered to be a mask.
        Default : False
    prior_inpainting : torch.Tensor
        Dictionary for inpainting.

    Returns
    -------
    out : torch.Tensor
        Next iterate
    """
    # Gradient descent
    if inpainting:
        approx = operator * torch.matmul(prior_inpainting, iterate)
        gradient = torch.matmul(prior_inpainting.t(), approx - operator * y)
    else:
        gradient = torch.matmul(
            operator.t(), torch.matmul(operator, iterate) - y
        )
    out = iterate - step * gradient

    # Thresholding
    thresh = torch.abs(out) - step * lambd
    out = torch.sign(out) * F.relu(thresh)

    return out


def fista_iteration(out, iterate_old, y, operator, step, t, lambd,
                    inpainting=False, prior_inpainting=None):
    """
    FISTA iteration

    Parameters
    ----------
    out : torch.Tensor
        Current iterate
    iterate_old: torch.Tensor
        Last point for momentum
    y : torch.Tensor
        data
    operator : torch.Tensor
        operator
    step : torch.Tensor, shape (1)
        Step size
    t : float
        FISTA momentum term
    lambd : float
        Regularization parameter
    inpainting : bool
        If set to True, the operator is
        considered to be a mask.
        Default : False
    prior_inpainting : torch.Tensor
        Dictionary for inpainting.

    Returns
    -------
    out : torch.Tensor
        Next iterate
    iterate : torch.Tensor
        Current point for momentum
    t_new : float
        Next momentum term
    """
    # Gradient descent
    if inpainting:
        approx = operator * torch.matmul(prior_inpainting, out)
        gradient = torch.matmul(prior_inpainting.t(), approx - operator * y)
    else:
        gradient = torch.matmul(
            operator.t(), torch.matmul(operator, out) - y
        )
    iterate = out - step * gradient

    # Thresholding
    thresh = torch.abs(iterate) - step * lambd
    iterate = torch.sign(iterate) * F.relu(thresh)

    # Momentum
    t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
    out = iterate + ((t-1) / t_new) * (iterate - iterate_old)

    return out, iterate, t_new


def condat_vu_iteration(iterate_p, iterate_d, y, operator,
                        prior, sigma, tau, lambd, inpainting=False):
    """
    Condat Vu iteration

    Parameters
    ----------
    iterate_p : torch.Tensor
        Current primal iterate
    iterate_d : torch.Tensor
        Current dual iterate
    y : torch.Tensor
        Data
    operator : torch.Tensor
        Measurement matrix
    prior : torch.Tensor
        Prior
    sigma : torch.Tensor, shape (1)
        Step size
    tau : torch.Tensor, shape (1)
        Step size
    lambd : float
        Regularization parameter

    Returns
    -------
    out_p : torch.Tensor
        Next primal iterate
    out_d : torch.Tensor
        Next dual iterate
    """
    # Gradient descent primal
    if inpainting:
        first_gradient = operator * (iterate_p - y)
    else:
        first_gradient = torch.matmul(
            operator.t(), torch.matmul(operator, iterate_p) - y
            )
    second_gradient = torch.matmul(prior.t(), iterate_d)
    out_p = iterate_p - tau * (first_gradient + second_gradient)

    # Gradient ascent dual
    gradient_dual = torch.matmul(prior, 2 * out_p - iterate_p)

    ascent = iterate_d + sigma * gradient_dual
    out_d = ascent / torch.max(torch.ones_like(ascent),
                               torch.abs(ascent) / lambd)
    return out_p, out_d


def lasso(y, x, z, operator, lambd, inpainting=False):
    """
    Lasso cost function

    Parameters
    ----------
    y : torch.Tensor
        Data
    x : torch.Tensor
        Recovered signal
    z : torch.Tensor
        Sparse representation/code
    operator : torch.Tensor
        Measurement matrix
    lambd : float
        Regularization parameter
    inpainting : bool
        If set to True, the operator is
        considered to be a mask.
        Default : False

    Returns
    -------
    float
        Loss value
    """
    if inpainting:
        product = operator * x
    else:
        product = torch.matmul(operator, x)
    res = product - y
    l2 = (res * res).sum()
    l1 = torch.abs(z).sum()

    return 0.5 * l2 + lambd * l1
