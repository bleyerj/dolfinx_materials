import numpy as np
from .elasticity import LinearElasticIsotropic
from .tensors import Identity, K
from scipy.optimize import fsolve


class ElastoPlasticIsotropicHardening:
    def __init__(self, elastic_model, yield_stress):
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        # self.hardening_modulus = hardening_modulus
        # self.hardening = lambda p: self.hardening_modulus * p

    def integrate(self, eps, state):
        eps_old = state["eps"]
        deps = eps - eps_old
        epsp_old = state["eps_p"]
        p_old = state["p"]
        sig_old = state["sig"]

        lmbda, mu = self.elastic_model.get_Lame_parameters()
        C = self.elastic_model.C
        sig_el = sig_old + C @ deps
        s_el = K() @ sig_el
        print("p_old", p_old)
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = np.sqrt(3 / 2.0) * np.linalg.norm(s_el)
        yield_criterion = sig_eq_el - sig_Y_old
        print("Yield:", yield_criterion)
        if yield_criterion > 0:
            dp = fsolve(
                lambda dp: sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp), 0.0
            )
            print(dp)
            # dp = yield_criterion / (3 * mu + self.hardening_modulus)
            depsp = 3 / 2 / sig_eq_el * s_el * dp
        else:
            dp = 0
            depsp = 0 * s_el
        sig = sig_el - 2 * mu * K() @ depsp
        print("New sig", sig)
        new_state = state.copy()
        new_state["eps"] = eps_old + deps
        new_state["eps_p"] = epsp_old + depsp
        new_state["p"] = p_old + dp
        new_state["sig"] = sig
        return sig, C, new_state