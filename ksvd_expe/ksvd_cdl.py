import time

import numpy as np
import pandas as pd
import scipy
import numba


@numba.jit((numba.float64[:, :], numba.float64[:, :]), cache=True)
def compute_DtD(u, v):  # pragma: no cover
    n_atoms, n_times_atom = v.shape

    DtD = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for t in range(n_times_atom):
                if t == 0:
                    DtD[k0, k, t0] = np.dot(v[k0], v[k])
                else:
                    DtD[k0, k, t0 + t] = np.dot(v[k0, :-t], v[k, t:])
                    DtD[k0, k, t0 - t] = np.dot(v[k0, t:], v[k, :-t])
    DtD *= np.dot(u, u.T).reshape(n_atoms, n_atoms, 1)
    return DtD


class KSVD_CDL():
    """
    K-SVD for convolutional dictionary laerning with one rank constraint.

    Parameters
    ----------
    n_atoms : int
        Number of atoms
    n_times_atom : int
        Size of the convolutional kernels
    n_iter : int, optional
        Number of iterations, default to 10
    tol : float, optional
        Threshold for sparse coding error, default to 0.1
    max_iter : int, optional
        Maximum number of sparse coding iterations, should not be reached if
        tol is reached before.
    correlation : str {'full', 'fast', 'debug'}
        Method to update the correlation in the sparse coding. If full, update
        the residual and recompute the full correlation each time, as done in
        the original algorithm. If fast, update directly the correlation using
        DtD. If 'debug', check that the two methods match.
    """
    def __init__(self, n_atoms, n_times_atom, n_iter=10, tol=0.1,
                 max_iter=100000, correlation='fast'):
        self.n_atoms = n_atoms
        self.n_times_atom = n_times_atom
        self.n_iter = n_iter
        self.tol = tol
        self.max_iter = max_iter
        self.correlation = correlation

    def multichannel_correlation(self, signals, u, v):
        """
        Compute the correlation between a signal and a bank of atoms.

        Note that this function considers that the atoms are rank1 u_k v_k^T
        and with global unit norms ||u_k v_k^T||_2 = 1.

        Parameters
        ----------
        signals : np.array(n_channels, n_times)
            multivariate signal
        u : np.array(n_atoms, n_channels)
            spatial map of the atoms
        v : np.array(n_atoms, n_times_atom)
            spatial map of the atoms

        Returns
        -------
        np.array(n_times - n_times_atom + 1)
            correlation
        """
        n_atoms = u.shape[0]
        assert v.shape[0] == n_atoms

        correlation = np.zeros((
            n_atoms, signals.shape[1] - self.n_times_atom + 1
        ))
        for k in range(n_atoms):
            correlation[k] = np.correlate(
                u[k] @ signals, v[k]
            )
        return correlation

    def csc(self, signals, u, v):
        """
        Convolutional sparse coding with rank one atoms

        Parameters
        ----------
        signals : np.array(n_channels, n_times)
            multivariate signals
        u : np.array(n_atoms, n_times_atom)
            spatial part of the dictionary
        v : np.array(n_atoms, n_channels)
            temporal part of the dictionary

        Returns
        -------
        index_times, coeffs, index_dico
            np.array, np.array, np.array
        """

        index_time = []
        coeffs = []
        index_dico = []

        # Compute correlation matrix
        R0 = signals.copy()
        Q = self.multichannel_correlation(R0, u, v)
        if self.correlation in ['debug', 'full']:
            current_D = self.get_D(u, v)
        if self.correlation in ['debug', 'fast']:
            DtD = compute_DtD(u, v)

        _, n_times_atom = v.shape
        _, n_times_valid = Q.shape

        for i in range(self.max_iter):
            if i % 50 == 0:
                print(f"Sparse coding - {i/self.max_iter:.1%}\r",
                      end='', flush=True)
            # Find max correlation
            index = np.argmax(abs(Q))
            k0, t0 = np.unravel_index(index, Q.shape)
            alphai = Q[k0, t0]

            if i == 0:
                alpha1 = alphai
            alpha0 = abs(alphai / alpha1)

            index_time.append(t0)
            coeffs.append(alphai)
            index_dico.append(k0)

            if self.correlation in ['debug', 'fast']:
                # define the bounds for the update of the correlation
                t_start_up = max(0, t0 - n_times_atom + 1)
                t_end_up = min(t0 + n_times_atom, n_times_valid)

                # update correlation
                ll = t_end_up - t_start_up
                off = max(0, n_times_atom - t0 - 1)
                Q[:, t_start_up:t_end_up] -= DtD[:, k0, off:off + ll] * alphai

            if self.correlation in ['debug', 'full']:

                # Update residual
                R0[:, t0:t0+self.n_times_atom] -= alphai * current_D[k0]

                # Update correlation
                Q0 = self.multichannel_correlation(R0, u, v)

                if self.correlation == 'debug':
                    if not np.allclose(Q, Q0):
                        import IPython
                        IPython.embed()
                        raise SystemExit(1)
                else:
                    Q[:] = Q0

            if alpha0 <= self.tol:

                print("Sparse coding - 'done (converged)'")
                break
        else:
            print("Sparse coding - done (reached max-iter)")

        return np.array(index_time), np.array(coeffs), np.array(index_dico)

    def get_uv(self, D):
        """
        From alphacsc
        Project D on the space of rank 1 dictionaries

        Parameter
        ---------
        D: array (n_atoms, n_channels, n_times_atom)
        Return
        ------
        uv: array (n_atoms, n_channels + n_times_atom)
        """
        n_atoms, n_channels, n_times_atom = D.shape
        u = np.zeros((n_atoms, n_channels))
        v = np.zeros((n_atoms, n_times_atom))
        for k, d in enumerate(D):
            U, s, V = scipy.sparse.linalg.svds(d, k=1)
            u[k], v[k] = U[:, 0], V[0]
        return u, v

    def get_D(self, u, v):
        """
        From alphacsc
        Compute the rank 1 dictionary associated with the given uv

        Parameter
        ---------
        uv: array (n_atoms, n_channels + n_times_atom)
        n_channels: int
            number of channels in the original multivariate series
        Return
        ------
        D: array (n_atoms, n_channels, n_times_atom)
        """

        return u[:, :, None] * v[:, None, :]

    def compute_residual(self, signal, index_dico, index_time, coeffs,
                         u, v, exclude=()):
        _, n_times_atom = v.shape
        Rk = signal.copy()
        for k0, t0, alpha_t in zip(index_dico, index_time, coeffs):
            if k0 not in exclude:
                # alpha_t * D0[k0]
                Rk[:, t0:t0+n_times_atom] -= (
                    alpha_t * u[k0, :, None] * v[k0, None, :]
                )
        return Rk

    def ksvd_cdl(self, signal, n_atoms, n_times_atom, n_iter):
        """
        K-SVD for CDL with rank one constraint

        Parameters
        ----------
        signal : np.array(n_channels, n_times)
            multivariate signal
        n_atoms : int
            number of components in the signal
        n_times_atom : int
            length of the convolutional kernel
        n_iter : int, optional
            number of iterations, by default 10

        Returns
        -------
        np.array(n_atoms, n_channels, n_n_times_atom)
            dictionary
        """

        n_channels, n_times = signal.shape

        # Init dictionary
        D0 = np.zeros((n_atoms, n_channels, n_times_atom))
        for i in range(n_atoms):
            k = np.random.randint(n_times - n_times_atom)
            D0[i, :] = signal[:, k:k+n_times_atom]
        norm = np.linalg.norm(D0, axis=(1, 2), keepdims=True)
        norm[np.where(np.isnan(norm))] = 1
        D0 /= np.linalg.norm(D0, axis=(1, 2), keepdims=True)

        u, v = self.get_uv(D0)

        t_start = time.perf_counter()

        pobj = []
        for j in range(n_iter):
            print('-' * 80 + f'\nIteration {j}')
            # CSC
            index_time, coeffs, index_dico = self.csc(signal, u, v)
            for k in range(n_atoms):
                print(f"Updating Dk - k={k}\r", end='', flush=True)
                # Update residual
                Rk = self.compute_residual(
                    signal, index_dico, index_time, coeffs, u, v, exclude=[k]
                )

                # Compute matrix for SVD
                A = []
                for k0, t0 in zip(index_dico, index_time):
                    if k0 == k:
                        A.append(Rk[:, t0:t0+n_times_atom].flatten())
                A = np.array(A)

                # SVD
                # *_, dk_flat = np.linalg.svd(A)
                *_, dk_flat = scipy.sparse.linalg.svds(A, k=1)

                # Update dico atom in rank1 shape
                dk = dk_flat[0].reshape((n_channels, -1))
                dk /= np.linalg.norm(dk, ord='fro')
                U, s, V = scipy.sparse.linalg.svds(dk, k=1)
                u[k], v[k] = U[:, 0], V[0]

                # Update coef with correlation
                for i, (k0, t0) in enumerate(zip(index_dico, index_time)):
                    if k0 == k:
                        coeffs[i] = (u[k] @ Rk[:, t0:t0+n_times_atom]) @ v[k]

            # Rank one constraint
            print('Updating D - projection...', end='', flush=True)
            print('done')

            # Display loss
            t = time.perf_counter() - t_start
            R = self.compute_residual(
                signal, index_dico, index_time, coeffs, u, v
            )
            loss = np.sum(R ** 2) / n_times
            pobj.append({'time': t, 'loss': loss})
            print(f'Final loss: {loss:.2e}')
            t_start = time.perf_counter()

        # sort the atoms by explained variance:
        index_time, coeffs, index_dico = self.csc(signal, u, v)
        variance = []
        for k in range(n_atoms):
            d_k = [i for i in range(n_atoms) if i != k]
            R_k = self.compute_residual(
                0 * signal, index_dico, index_time, coeffs, u, v,
                exclude=d_k
            )
            variance.append(R_k.var())
        i0 = np.array(variance).argsort()
        u, v = u[i0], v[i0]

        return u, v, pd.DataFrame(pobj)

    def fit(self, X, y=None):
        X = X / X.std()
        self.u_hat_, self.v_hat_, self.pobj_ = self.ksvd_cdl(
            X, self.n_atoms, self.n_times_atom, self.n_iter
        )
