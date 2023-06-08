import numpy as np
from dolfinx_materials.material import Material


class LinearElasticIsotropic(Material):
    def get_Lame_parameters(self, E, nu):
        return E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)

    def compute_C(self, E, nu):
        lmbda, mu = self.get_Lame_parameters(E, nu)
        C = 2 * mu * np.eye(6)
        C[:3, :3] += lmbda
        return C

    def constitutive_update(self, eps, state):
        E = self.material_properties["E"]
        nu = self.material_properties["nu"]
        C = self.compute_C(E, nu)
        return np.dot(self.C, eps), self.C


class PlaneStressLinearElasticIsotropic(LinearElasticIsotropic):
    def compute_C(self, E, nu):
        C = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, 1 - nu]])
        return C
