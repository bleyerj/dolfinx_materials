import numpy as np


class LinearElasticIsotropic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.C = self.compute_C(E, nu)

    def get_Lame_parameters(self):
        lmbda = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        mu = self.E / 2 / (1 - 2 * self.nu)
        return lmbda, mu

    def compute_C(self, E, nu):
        lmbda, mu = self.get_Lame_parameters()
        C = 2 * mu * np.eye(6)
        C[:3, :3] += lmbda
        return C

    def integrate(self, eps, state):
        return np.dot(self.C, eps), self.C, state
    
class PlaneStressLinearElasticIsotropic:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.C = self.compute_C(E, nu)

    def get_Lame_parameters(self):
        lmbda = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        mu = self.E / 2 / (1 - 2 * self.nu)
        return lmbda, mu

    def compute_C(self, E, nu):
        C = E/(1-nu**2)*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, 1-nu]])
        return C

    def integrate(self, eps, state):
        return np.dot(self.C, eps), self.C, state
