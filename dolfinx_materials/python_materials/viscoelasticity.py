import numpy as np
from dolfinx_materials.material import Material
from dolfinx_materials.material.generic import tangent_AD
from .elasticity import LinearElasticIsotropic
import jax.numpy as jnp
import jax


class LinearViscoElasticity(Material):

    def __init__(self, branch0, branch1, eta, nud):
        super().__init__()
        self.branch0 = branch0  # should be a LinearElastic material
        self.branch1 = branch1  # should be a LinearElastic material
        self.eta = eta
        self.nud = nud  # dissipative Poisson ratio
        self.Cd = LinearElasticIsotropic(self.eta, self.nud).C

    @property
    def internal_state_variables(self):
        return {"epsv": 6}

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        epsv_old = state["epsv"]
        Id = np.eye(6)
        iTau = self.branch1.C @ np.linalg.inv(self.Cd)
        A = jnp.linalg.inv(Id + dt * iTau)
        epsv_new = A @ (epsv_old + dt * iTau @ eps)
        sig = self.branch0.C @ eps + self.branch1.C @ (eps - epsv_new)
        state["epsv"] = epsv_new
        state["Strain"] = eps
        state["Stress"] = sig
        return sig, state
