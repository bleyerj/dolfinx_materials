import numpy as np
from .elasticity import LinearElasticIsotropic


class LinearViscoElasticity:
    def __init__(self, branch1, branch2, eta, nud, dt):
        self.branch1 = branch1  # should be a LinearElastic material
        self.branch2 = branch2  # should be a LinearElastic material
        self.eta = eta
        self.nud = nud  # dissipative Poisson ratio
        self.dt = dt
        self.Cd = LinearElasticIsotropic(self.eta, self.nud).C

    def integrate(self, eps, state):
        epsv_old = state["epsv"]
        Id = np.eye(6)
        A = np.linalg.inv(Id + self.dt * self.Cd)
        epsv_new = A @ (epsv_old + self.dt * self.Cd @ eps)
        sig = self.branch1.C @ eps + self.branch2.C @ (eps - epsv_new)
        Ct = self.branch1.C + self.branch2.C @ (Id - self.dt * A @ self.Cd)

        new_state = state.copy()
        new_state["epsv"] = epsv_new
        return sig, Ct, new_state
