from dolfinx_materials.material import Material
import cvxpy as cp
import numpy as np


class NielsenPlate(Material):
    def __init__(self, elastic_model, thick, yield_parameters):
        super().__init__()
        self.elastic_model = elastic_model
        self.thickness = thick
        self.yield_parameters = yield_parameters
        self.set_cvxpy_model()

    def set_cvxpy_model(self):
        self.M = cp.Variable((3,))
        self.M_el = cp.Parameter((3,))
        t = cp.Variable()
        self.normalization = cp.Parameter()
        obj = 0.5 * cp.quad_form(
            self.M - self.M_el, np.linalg.inv(self.elastic_model.C)
        )
        yp = self.yield_parameters
        Mxx = self.M[0]
        Myy = self.M[1]
        Mxy = self.M[2] / np.sqrt(2)
        cons = [
            cp.quad_over_lin(Mxy, yp["mxp"] - Mxx) <= yp["myp"] - Myy,
            cp.quad_over_lin(Mxy, yp["mxm"] + Mxx) <= yp["mym"] + Myy,
            obj <= self.normalization * t,
        ]
        cons = [obj <= self.normalization * t, cp.norm(self.M) <= yp["mxp"]]
        self.prob = cp.Problem(cp.Minimize(t), cons)
        # self.prob = cp.Problem(cp.Minimize(obj))

    @property
    def gradients(self):
        return {"curv": 3}

    @property
    def fluxes(self):
        return {"bending": 3}

    def constitutive_update(self, χ, state):
        χ_old = state["curv"]
        dχ = χ - χ_old
        M_old = state["bending"]

        D = self.thickness**3 / 12 * self.elastic_model.C
        S = np.linalg.inv(self.elastic_model.C)
        M_pred = M_old + D @ dχ
        self.M_el.value = M_pred
        self.normalization.value = max(M_pred @ S @ M_pred, 1e-4)

        self.prob.solve(
            # requires_grad=True,
            solver=cp.ECOS,
            verbose=False,
            # eps_abs=1e-6,
            # eps_rel=1e-6,
        )
        M = self.M.value

        # assert np.allclose(M, self.M_el.value)
        # It = np.zeros((3, 3))
        # for i in range(3):
        #     e = np.zeros((3,))
        #     e[i] = 1
        #     self.M_el.delta = e
        #     self.prob.derivative()
        #     It[:, i] = self.M.delta

        state["curv"] = χ
        state["bending"] = M
        return M, D  # @ It
