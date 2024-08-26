import numpy as np
from dolfinx_materials.jax_materials import (
    vonMisesIsotropicHardening,
    LinearElasticIsotropic,
    LinearViscoElasticity,
    GeneralizedMaxwell,
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

    b = 1e3
    sigu = 750.0

    def yield_stress(p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))

    material = vonMisesIsotropicHardening(elastic_model, yield_stress)
    eps = 2e-2
    Eps = np.vstack(([0, 0, 0, np.sqrt(2) * eps, 0, 0],) * Nbatch)
    material.set_data_manager(Nbatch)
    state = material.get_initial_state_dict()

    plt.figure()
    for t in np.linspace(0, 1.0, 50):
        with Timer("Integration"):
            sig, isv, Ct = material.integrate(t * Eps)
        material.data_manager.update()
        plt.scatter(eps * t, sig[0, 3] / np.sqrt(2), color="b")
    while sig[0, 3] > 0:
        t -= 1 / 50.0
        sig, isv, Ct = material.integrate(t * Eps)
        material.data_manager.update()
        if sig[0, 3] < 0:
            break
        else:
            plt.scatter(eps * t, sig[0, 3] / np.sqrt(2), color="b")
    plt.xlabel(r"Shear strain $\varepsilon_{xy}$")
    plt.ylabel(r"Shear stress $\sigma_{xy}$ [MPa]")
    plt.savefig("elastoplastic_shear_1d.svg")
    plt.show()


from mpi4py import MPI
from dolfinx.common import list_timings, TimingType, Timer


def test_elastic(Nbatch):
    E = 10.0
    nu = 0.3
    material = LinearElasticIsotropic(E, nu)

    eps = [1e-3, 0, 0, 0, 0, 0]
    Eps = np.vstack((eps,) * Nbatch)
    material.set_data_manager(Nbatch)
    state = material.get_initial_state_dict()

    dt = 0
    sig, state, Ct = material.integrate(Eps, dt)
    assert np.allclose(sig, Eps @ material.C)


def test_viscoelastic(Nbatch):
    times = np.linspace(0, 0.5, 100)
    dt = np.diff(times)
    E0 = 70e3
    E1 = 20e3
    nu = 0.0
    tau = 0.05
    branch0 = LinearElasticIsotropic(E0, nu)
    branch1 = LinearElasticIsotropic(E1, nu)
    material = LinearViscoElasticity(branch0, branch1, tau, nu)

    K0 = E0 / (3 * (1 - 2 * nu))
    mu0 = E0 / 2 / (1 + nu)
    K1 = E1 / (3 * (1 - 2 * nu))
    mu1 = E1 / 2 / (1 + nu)

    material = GeneralizedMaxwell(
        K0, mu0, [K1 / 2, K1 / 2], [mu1 / 2, mu1 / 2], [tau, tau]
    )

    material = GeneralizedMaxwell(K0, mu0, K1, mu1, tau)
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
        print(sig)

        sig_th = E0 * epsr + E1 * epsr * np.exp(-t / tau)
        material.data_manager.update()
        plt.scatter(t, sig[0, 0], color="b")
        plt.scatter(t, sig_th, color="r")
    plt.show()


test_elastoplastic(1)

# test_elastic(10)
