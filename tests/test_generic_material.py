import numpy as np
from dolfinx_materials.python_materials import (
    ElastoPlasticIsotropicHardening,
    LinearElasticIsotropic,
    LinearViscoElasticity,
)
import matplotlib.pyplot as plt


def test_elastoplastic(Nbatch):
    E = 70e3
    nu = 0.3
    H = E / 10.0
    sig0 = 500.0
    elastic_model = LinearElasticIsotropic(E, nu)
    mu = E / 2 / (1 + nu)

    def yield_stress(p):
        return sig0 + H * p

    material = ElastoPlasticIsotropicHardening(elastic_model, yield_stress)
    eps = 2e-2
    Eps = np.vstack(([0, 0, 0, np.sqrt(2) * eps, 0, 0],) * Nbatch)
    material.set_data_manager(Nbatch)
    state = material.get_initial_state_dict()

    plt.figure()
    for t in np.linspace(0, 1.0, 20):
        G = 2 * mu
        GH = H / np.sqrt(3)
        sig_th = G * eps * t
        k0 = sig0 / np.sqrt(3)
        if abs(sig_th) > k0:
            sig_th = k0 + G * GH / (G + GH) * (eps * t - k0 / G)
        sig, state, Ct = material.integrate(t * Eps)
        material.data_manager.update()
        plt.scatter(eps * t, sig[0, 3] / np.sqrt(2), color="b")
        plt.scatter(eps * t, sig_th, color="r")

    plt.show()


def test_viscoelastic(Nbatch):
    times = np.linspace(0, 0.1, 10) ** 0.5
    dt = 1 / len(times)
    E0 = 70e3
    E1 = 20e3
    nu = 0.3
    eta = 1e3
    branch0 = LinearElasticIsotropic(E0, nu)
    branch1 = LinearElasticIsotropic(E1, nu)
    material = LinearViscoElasticity(branch0, branch1, eta, nu)

    epsr = 1e-3
    Eps = np.vstack(([epsr, 0, 0, 0, 0, 0],) * Nbatch)
    material.set_data_manager(Nbatch)
    state = material.get_initial_state_dict()
    plt.figure()
    for t in times:
        dt *= 2
        sig, state, Ct = material.integrate(Eps, dt)
        print(Ct[0, 0, 0])

        material.data_manager.update()
        plt.scatter(t, sig[0, 0], color="b")
        # plt.scatter(times, sig_th, color="r")
    plt.show()


test_elastoplastic(2)

# test_viscoelastic(2)
