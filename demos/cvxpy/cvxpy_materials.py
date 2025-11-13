from dolfinx_materials.material import Material
import cvxpy as cp
import numpy as np


class CvxPyMaterial(Material):
    def __init__(self, E, nu, **kwargs):
        super().__init__()
        self.E, self.nu = E, nu

        # Handle any additional keyword arguments passed
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Plane stress stiffness matrix (Kelvin–Mandel form)
        self.C = (self.E / (1 - self.nu**2)) * np.array(
            [[1.0, self.nu, 0.0], [self.nu, 1.0, 0.0], [0.0, 0.0, (1.0 - self.nu)]]
        )
        self.set_cvxpy_model()

    @property
    def gradients(self):
        return {"Strain": 3}

    @property
    def fluxes(self):
        return {"Stress": 3}

    def yield_constraints(self, Sig):
        return []

    def set_cvxpy_model(self):
        self.sig = cp.Variable((3,))
        self.sig_el = cp.Parameter((3,))
        obj = 0.5 * cp.quad_form(self.sig - self.sig_el, np.linalg.inv(self.C))
        self.prob = cp.Problem(cp.Minimize(obj), self.yield_constraints(self.sig))

    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]
        deps = eps - eps_old
        sig_old = state["Stress"]

        sig_pred = sig_old + self.C @ deps

        self.sig_el.value = sig_pred
        self.prob.solve()
        sig = self.sig.value

        state["Strain"] = eps
        state["Stress"] = sig
        return self.C, state


class Rankine(CvxPyMaterial):
    def yield_constraints(self, sig):
        Sig = cp.bmat(
            [
                [sig[0], sig[2] / np.sqrt(2)],
                [sig[2] / np.sqrt(2), sig[1]],
            ]
        )
        return [
            cp.lambda_max(Sig) <= self.ft,
            cp.lambda_min(Sig) >= -self.fc,
        ]


class L1Rankine(CvxPyMaterial):

    def yield_constraints(self, sig):
        Sig = cp.bmat(
            [
                [sig[0], sig[2] / np.sqrt(2)],
                [sig[2] / np.sqrt(2), sig[1]],
            ]
        )
        ft = self.ft
        fc = self.fc
        s = cp.vstack([Sig[0, 0] - Sig[1, 1], 2 * Sig[0, 1]])
        R = cp.norm(s)
        T = cp.trace(Sig)
        return [
            T <= ft,
            T >= -fc,
            T * (1 / ft - 1 / fc) / 2 + R * (1 / ft + 1 / fc) / 2 <= 1,
        ]


class PlaneStressvonMises(CvxPyMaterial):
    def yield_constraints(self, sig):
        Q = np.array([[1, -1 / 2, 0], [-1 / 2, 1, 0], [0, 0, 1]])
        sig_eq2 = cp.quad_form(sig, Q)
        return [sig_eq2 <= self.sig0**2]


class PlaneStressHosford(CvxPyMaterial):
    def yield_constraints(self, sig):
        Sig = cp.bmat(
            [
                [sig[0], sig[2] / np.sqrt(2)],
                [sig[2] / np.sqrt(2), sig[1]],
            ]
        )
        z = cp.Variable(3)
        return [
            cp.trace(Sig) == z[0] - z[1],
            cp.lambda_max(Sig) - cp.lambda_min(Sig) <= z[2],
            z[2] == z[0] + z[1],
            cp.norm(z, p=self.a) <= 2 ** (1 / self.a) * self.sig0,
        ]
