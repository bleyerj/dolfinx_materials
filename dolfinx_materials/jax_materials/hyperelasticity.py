import jax
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD
from .tensors import to_mat, to_vect


class SaintVenantKirchhoff(JAXMaterial):
    def __init__(self, elastic_model):
        super().__init__()
        self.elastic_model = elastic_model

    @property
    def fluxes(self):
        return {"PK1": 9}

    @property
    def gradients(self):
        return {"F": 9}

    @tangent_AD
    def constitutive_update(self, Fv, state, dt):
        C = self.elastic_model.C
        I = jnp.eye(3)

        def free_energy(Fv):
            F = to_mat(Fv)
            Ev = to_vect(0.5 * (F.T @ F - I), symmetry=True)
            return 0.5 * Ev @ C @ Ev

        P = jax.jacfwd(free_energy, Fv)
        state["F"] = Fv
        state["PK1"] = P

        return P, state
