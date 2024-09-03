import numpy as np
from dolfinx_materials.jax_materials import (
    LinearElasticIsotropic,
)
from dolfinx_materials.jax_materials.tensors import to_mat, to_vect
from dolfinx_materials.jax_materials.finite_strain_viscoplasticity import (
    FeFpJ2Plasticity,
)
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from time import time


def test_FeFp_plasticity(Nbatch):
    E = 70e3
    nu = 0.3
    sig0 = 500.0

    b = 10
    sigu = 750.0

    def yield_stress(p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))

    elastic_model = LinearElasticIsotropic(E, nu)

    # material = vonMisesIsotropicHardening(elastic_model, yield_stress)
    material = FeFpJ2Plasticity(elastic_model, yield_stress, theta=0.5)
    eps = 2e-2
    F0 = np.vstack(([1 + eps, 1, 1, 0, 0, 0, 0, 0, 0],) * Nbatch)
    material.set_data_manager(Nbatch)
    state = material.get_initial_state_dict()

    def initialize(state):
        I = to_vect(jnp.eye(3))
        state["F"] = I
        state["be_bar"] = to_vect(jnp.eye(3), True)
        state["Fe"] = I
        # state["Fp"] = I
        return state

    state = jax.vmap(initialize)(state)
    material.set_initial_state_dict(state)

    # F = np.array(F0)
    # F[:, 0] = 1 + 1e-6
    # P, isv, Ct = material.integrate(F)
    # print(P)

    plt.figure()
    Nsteps = 20
    dt = 0
    for t in np.linspace(0, 1.0, Nsteps)[1:]:
        # with Timer("Integration"):
        #     sig, isv, Ct = material.integrate(t * Eps)
        tic = time()
        F = np.array(F0)
        F[:, 0] = 1 + eps * t
        P, isv, Ct = material.integrate(F, dt)
        print(P[0])
        plt.scatter(eps * t, P[0][0], color="b")
        material.data_manager.update()

    plt.show()


test_FeFp_plasticity(1000)
