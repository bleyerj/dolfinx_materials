import jax.numpy as jnp
from dolfinx_materials.jaxmat import JAXMaterial
import jaxmat.materials as jm


def test_FeFp_plasticity(Nbatch=10):
    E = 70e3
    nu = 0.3
    sig0 = 500.0

    b = 1000
    sigu = 750.0

    def yield_stress(p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))

    elastic_model = jm.LinearElasticIsotropic(E=E, nu=nu)

    behavior = jm.FeFpJ2Plasticity(elasticity=elastic_model, yield_stress=yield_stress)
    material = JAXMaterial(behavior)
    material.set_data_manager(Nbatch)

    eps = 2e-2

    Nsteps = 20
    dt = 0
    for t in jnp.linspace(0, 1.0, Nsteps)[1:]:
        F = jnp.zeros((Nbatch, 9))
        F = F.at[:, 0].set(1 + eps * t)
        F = F.at[:, [1, 2]].set(1 - eps / 2 * t)
        P, isv, Ct = material.integrate(F, dt)

        material.data_manager.update()
