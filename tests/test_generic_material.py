import numpy as np
from dolfinx_materials.python_materials import (
    ElastoPlasticIsotropicHardening,
    LinearElasticIsotropic,
    LinearViscoElasticity,
)
import matplotlib.pyplot as plt
import jax.numpy as jnp


def test_elastoplastic(Nbatch):
    E = 70e3
    nu = 0.3
    H = E / 10.0
    sig0 = 500.0
    elastic_model = LinearElasticIsotropic(E, nu)
    mu = E / 2 / (1 + nu)

    # def yield_stress(p):
    #     return sig0 + H * p

    b = 1e3
    sigu = 750.0

    def yield_stress(p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))

    material = ElastoPlasticIsotropicHardening(elastic_model, yield_stress)
    eps = 2e-2
    Eps = np.vstack(([0, 0, 0, np.sqrt(2) * eps, 0, 0],) * Nbatch)
    material.set_data_manager(Nbatch)
    state = material.get_initial_state_dict()

    plt.figure()
    for t in np.linspace(0, 1.0, 50):
        # print(t)
        G = 2 * mu
        GH = H / np.sqrt(3)
        sig_th = G * eps * t
        k0 = sig0 / np.sqrt(3)
        if abs(sig_th) > k0:
            sig_th = k0 + G * GH / (G + GH) * (eps * t - k0 / G)
        with Timer("Integration"):
            sig, state, Ct = material.integrate(t * Eps)

        material.data_manager.update()
        plt.scatter(eps * t, sig[0, 3] / np.sqrt(2), color="b")
        # plt.scatter(eps * t, sig_th, color="r")

    plt.show()


from mpi4py import MPI
from dolfinx.common import list_timings, TimingType, Timer


def test_viscoelastic(Nbatch):
    times = np.linspace(0, 0.5, 10)
    dt = np.diff(times)
    E0 = 70e3
    E1 = 20e3
    nu = 0.0
    eta = 1e3
    tau = eta / E1
    branch0 = LinearElasticIsotropic(E0, nu)
    branch1 = LinearElasticIsotropic(E1, nu)
    material = LinearViscoElasticity(branch0, branch1, eta, nu)

    epsr = 1e-3
    Eps = np.vstack(([epsr, 0, 0, 0, 0, 0],) * Nbatch)
    material.set_data_manager(Nbatch)
    state = material.get_initial_state_dict()
    t = 0

    plt.figure()
    for dti in dt:
        t += dti
        with Timer("Integration"):
            sig, state, Ct = material.integrate(Eps, dti)
        print(Ct[0, 0, 0])

        sig_th = E0 * epsr + E1 * epsr * np.exp(-t / tau)
        material.data_manager.update()
    #     plt.scatter(t, sig[0, 0], color="b")
    #     plt.scatter(t, sig_th, color="r")
    # plt.show()


test_elastoplastic(10000)

# test_viscoelastic(int(2000))

print(list_timings(MPI.COMM_WORLD, [TimingType.wall, TimingType.user]))
