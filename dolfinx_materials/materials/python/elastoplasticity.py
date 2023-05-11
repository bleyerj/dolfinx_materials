import numpy as np
from .elasticity import LinearElasticIsotropic
from .python_material import PythonMaterial
from .tensors import Identity, K
from scipy.optimize import fsolve


class ElastoPlasticIsotropicHardening(PythonMaterial):
    def __init__(self, elastic_model, yield_stress):
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        # self.hardening_modulus = hardening_modulus
        # self.hardening = lambda p: self.hardening_modulus * p
    
    def get_internal_state_variables(self):
        return {"p": 1}


    def constitutive_update(self, eps, state):
        eps_old = state["eps"]
        deps = eps - eps_old
        p_old = state["p"]
        sig_old = state["sig"]
        
        lmbda, mu = self.elastic_model.get_Lame_parameters()
        C = self.elastic_model.C
        sig_el = sig_old + C @ deps
        s_el = K() @ sig_el
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = np.sqrt(3 / 2.0) * np.linalg.norm(s_el)
        yield_criterion = sig_eq_el - sig_Y_old
        if yield_criterion >= 0:
            dp = fsolve(
                lambda dp: sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp), 0.0
            )
            # dp = yield_criterion / (3 * mu + self.hardening_modulus)
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
