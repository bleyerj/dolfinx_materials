from dolfinx_materials.material import Material
import cvxpy as cp
import numpy as np


class Rankine(Material):
    def __init__(self, elastic_model, fc, ft):
        super().__init__()
        self.elastic_model = elastic_model
        self.fc = fc
        self.ft = ft
        self.set_cvxpy_model()

    @property
    def gradients(self):
        return {"Strain": 3}

    @property
    def fluxes(self):
        return {"Stress": 3}

    def set_cvxpy_model(self):
        self.sig = cp.Variable((3,))
        self.sig_el = cp.Parameter((3,))
        obj = 0.5 * cp.quad_form(
            self.sig - self.sig_el, np.linalg.inv(self.elastic_model.C)
        )
        t = cp.Variable()
        Sig = cp.bmat(
            [
                [self.sig[0], self.sig[2] / np.sqrt(2)],
                [self.sig[2] / np.sqrt(2), self.sig[1]],
            ]
        )
        cons = [
            cp.lambda_max(Sig) <= self.ft,
            cp.lambda_min(Sig) >= -self.fc,
            obj <= t,
        ]
        self.prob = cp.Problem(cp.Minimize(t), cons)

    def constitutive_update(self, eps, state):
        eps_old = state["Strain"]
        deps = eps - eps_old
        sig_old = state["Stress"]

        sig_pred = sig_old + self.elastic_model.C @ deps

        self.sig_el.value = sig_pred
        self.prob.solve(
            solver=cp.MOSEK,
            verbose=False,
        )
        sig = self.sig.value

        state["Strain"] = eps
        state["Stress"] = sig
        return sig, self.elastic_model.C
