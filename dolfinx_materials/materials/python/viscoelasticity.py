import numpy as np
from .elasticity import LinearElasticIsotropic


class LinearViscoElasticity:
    def __init__(self, branch1, branch2, eta, nud, dt=0):
        self.branch1 = branch1  # should be a LinearElastic material
        self.branch2 = branch2  # should be a LinearElastic material
        self.eta = eta
        self.nud = nud  # dissipative Poisson ratio
        self.dt = dt
        self.Cd = LinearElasticIsotropic(self.eta, self.nud).C

    def integrate(self, eps, state):
        epsv_old = state["epsv"]
        Id = np.eye(6)
        iTau = self.branch2.C @ np.linalg.inv(self.Cd)
        A = np.linalg.inv(Id + self.dt * iTau)
        epsv_new = A @ (epsv_old + self.dt * iTau @ eps)
        sig = self.branch1.C @ eps + self.branch2.C @ (eps - epsv_new)
        Ct = self.branch1.C + self.branch2.C @ (Id - self.dt * A @ iTau)

        new_state = state.copy()
        new_state["epsv"] = epsv_new
        return sig, Ct, new_state
