import numpy as np
from .tensors import Identity, K, J
from scipy.optimize import fsolve


class RambergOsgood:
    def __init__(self, E, nu, sig0, n, alpha):
        self.E = E
        self.nu = nu
        self.sig0 = sig0
        self.n = n
        self.alpha = alpha

    def integrate(self, eps, state):
        eps_old = state["eps"]
        sig_old = state["sig"]
        print("eps", eps)
        ed = K() @ eps
        eps_eq = np.sqrt(2 / 3.0) * np.linalg.norm(ed)
        n_e = 2 / 3.0 * ed / max(eps_eq, 1e-10)

        mu = self.E / 2 / (1 + self.nu)
        kappa = self.E / 3 / (1 - 2 * self.nu)

        print(eps_eq)
        sig_eq = self.sig0 * (eps_eq / self.alpha) ** (1.0 / self.n)
        e = self.sig0 * (eps_eq / self.alpha) ** (1.0 / self.n)
        print(sig_eq / 3 / mu + self.alpha * (sig_eq / self.sig0) ** self.n)

        def residual(sig_eq):
            return (
                sig_eq / 3 / mu + self.alpha * (sig_eq / self.sig0) ** self.n
            ) - eps_eq

        sig_eq = fsolve(residual, self.sig0 * (eps_eq / self.alpha) ** (1.0 / self.n))
        assert sig_eq >= 0
        print("sig_eq", sig_eq)
        JJ = J()
        eps_m = J() @ eps
        sig = 3 * kappa * eps_m + 2 * sig_eq / 3 * n_e
        print("sigxx", sig[0])
        C = 3 * kappa * J() + 2 * mu * K()
        new_state = state.copy()
        new_state["eps"] = eps
        new_state["sig"] = sig
        return sig, C, new_state
