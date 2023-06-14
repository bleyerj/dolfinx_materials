import numpy as np
from dolfinx_materials.python_materials import (
    ElastoPlasticIsotropicHardening,
    LinearElasticIsotropic,
)
import matplotlib.pyplot as plt


def test_elastoplastic():
    E = 70e3
    nu = 0.0
    H = E / 1e3
    sig0 = 500.0
    elastic_model = LinearElasticIsotropic(E, nu)

    def yield_stress(p):
        return sig0 + H * p

    material = ElastoPlasticIsotropicHardening(
        elastic_model=elastic_model, yield_stress=yield_stress
    )

    eps = 2e-2
    Eps = np.array([[eps, 0, 0, 0, 0, 0]])
    material.set_data_manager(1)
    state = material.get_initial_state_dict()
    plt.figure()
    for t in np.linspace(0, 1.0, 20):
        sig_th = E * eps * t
        if abs(sig_th) > sig0:
            sig_th = sig0 + E * H / (E + H) * (eps * t - sig0 / E)
        sig, Ct = material.integrate(t * Eps)
        material.data_manager.update()
        plt.scatter(eps * t, sig[0, 0], color="b")
        plt.scatter(eps * t, sig_th, color="r")

    plt.show()


test_elastoplastic()
