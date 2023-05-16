import numpy as np
from .elasticity import LinearElasticIsotropic
from dolfinx_materials.material import Material
from .tensors import Identity, K
from scipy.optimize import fsolve


class ElastoPlasticIsotropicHardening(Material):
    def get_internal_state_variables(self):
        return {"p": 1}

    def constitutive_update(self, eps, state):
        eps_old = state["eps"]
        deps = eps - eps_old
        p_old = state["p"]
        sig_old = state["sig"]

        elastic_model = self.elastic_model
        E, nu = elastic_model.E, elastic_model.nu
        lmbda, mu = elastic_model.get_Lame_parameters(E, nu)
        C = self.elastic_model.compute_C(E, nu)
        sig_el = sig_old + C @ deps
        s_el = K() @ sig_el
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = np.sqrt(3 / 2.0) * np.linalg.norm(s_el)
        yield_criterion = sig_eq_el - sig_Y_old
        if yield_criterion >= 0:
            dp = fsolve(
                lambda dp: sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp), 0.0
            )
            n_el = s_el / sig_eq_el  # normal vector
            depsp = 3 / 2 * n_el * dp
            sig_Y_new = self.yield_stress(p_old + dp)
            dR_dp = (sig_Y_new - sig_Y_old) / dp
            beta = 1 - sig_Y_new / sig_eq_el
            gamma = 3 * mu / (3 * mu + dR_dp)
            D = 3 * mu * (gamma - beta) * np.outer(n_el, n_el) + 2 * mu * beta * K()
            Ct = C - D
        else:
            dp = 0
            depsp = 0 * s_el
            Ct = C
        sig = sig_el - 2 * mu * K() @ depsp

        state["eps"] = eps_old + deps
        state["p"] = p_old + dp
        state["sig"] = sig
        return sig, Ct
