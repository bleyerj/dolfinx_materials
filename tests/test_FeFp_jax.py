import numpy as np
from dolfinx_materials.jax_materials import (
    LinearElasticIsotropic,
)
from dolfinx_materials.jax_materials.tensors import to_vect
from dolfinx_materials.jax_materials.fe_fp_elastoplasticity import (
    FeFpJ2Plasticity,
)
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


def test_FeFp_plasticity(Nbatch=10):
    E = 70e3
    nu = 0.3
    sig0 = 500.0

    b = 10
    sigu = 750.0

    def yield_stress(p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))

    elastic_model = LinearElasticIsotropic(E, nu)

    material = FeFpJ2Plasticity(elastic_model, yield_stress, theta=0.5)
    eps = 2e-2
    F0 = np.vstack(([1 + eps, 1, 1, 0, 0, 0, 0, 0, 0],) * Nbatch)
    material.set_data_manager(Nbatch)
    state = material.get_initial_state_dict()

    def initialize(state):
        I = to_vect(jnp.eye(3))
        state["F"] = I
        state["be_bar"] = to_vect(jnp.eye(3), True)
        return state

    state = jax.vmap(initialize)(state)
    material.set_initial_state_dict(state)

    plt.figure()
    Nsteps = 20
    dt = 0
    for t in np.linspace(0, 1.0, Nsteps)[1:]:
        F = np.array(F0)
        F[:, 0] = 1 + eps * t
        P, isv, Ct = material.integrate(F, dt)
        plt.scatter(eps * t, P[0][0], color="b")
        material.data_manager.update()

    plt.show()
