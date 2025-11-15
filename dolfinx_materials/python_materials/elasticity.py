import numpy as np
from dolfinx_materials.generic import Material


class LinearElasticIsotropic(Material):
    def __init__(self, E, nu):
        super().__init__()
        self.E = E
        self.nu = nu
        self.C = self.compute_C(E, nu)

    def get_Lame_parameters(self, E, nu):
        return E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)

    def compute_C(self, E, nu):
        lmbda, mu = self.get_Lame_parameters(E, nu)
        C = 2 * mu * np.eye(6)
        C[:3, :3] += lmbda
        return C

    def constitutive_update(self, eps, state, dt):
        sig = np.dot(self.C, eps)
        state["Stress"] = sig
        return self.C, state
