import numpy as np


class LinearElasticIsotropic:
    def __init__(self, E, nu):
        self.C = self.compute_C(E, nu)

    def compute_C(self, E, nu):
        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 - 2 * nu)
        C = 2 * mu * np.eye(6)
        C[:3, :3] += lmbda
        return C

    def integrate(self, eps, state):
        return np.dot(self.C, eps), self.C, state
