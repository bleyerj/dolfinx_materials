import numpy as np
from .tensors import Identity, K, J
from scipy.optimize import fsolve
from .python_material import PythonMaterial

class RambergOsgood(PythonMaterial):
    def __init__(self, E, nu, sig0, n, alpha, eps_tol=1e-10):
        self.E = E
        self.nu = nu
        self.sig0 = sig0
        self.n = n
        self.alpha = alpha
        self.eps_tol = eps_tol

    def constitutive_update(self, eps, state):
        ed = K() @ eps
        eps_eq = np.sqrt(2 / 3.0) * np.linalg.norm(ed)

        mu = self.E / 2 / (1 + self.nu)
        kappa = self.E / 3 / (1 - 2 * self.nu)

        if eps_eq < self.eps_tol:
            C = 3 * kappa * J() + 2 * mu * K()
            sig = C @ eps
        else:
            n_e = 2 / 3.0 * ed / eps_eq
            sig_eq = self.sig0 * (eps_eq / self.alpha) ** (1.0 / self.n)
            sig_eq_ini = self.sig0 * (eps_eq / self.alpha) ** (1.0 / self.n)

            def f(sig_eq):
                return sig_eq / 3 / mu + self.alpha * (sig_eq / self.sig0) ** self.n

            sig_eq = fsolve(
                lambda sig_eq: f(sig_eq) - eps_eq,
                sig_eq_ini,
            )
            assert sig_eq >= 0

            eps_m = J() @ eps

            df_dsig_eq = (
                1 / 3 / mu
                + self.alpha * self.n * sig_eq ** (self.n - 1) / self.sig0**self.n
            )
            sig = 3 * kappa * eps_m + 2 * sig_eq / 3 * n_e
            dne_deps = 1 / eps_eq * (2 / 3.0 * K() - np.outer(n_e, n_e))

            C = (
                3 * kappa * J()
                + 1 / df_dsig_eq * np.outer(n_e, n_e)
                + sig_eq * dne_deps
            )

        state["eps"] = eps
        state["sig"] = sig
        return sig, C
