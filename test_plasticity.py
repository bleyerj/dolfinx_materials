import numpy as np
from dolfinx_materials.materials.python import (
    ElastoPlasticIsotropicHardening,
    LinearElasticIsotropic,
)
from uniaxial_test import uniaxial_test_2D

elastic_model = LinearElasticIsotropic(70e3, 0.3)
sig0 = 500.0
sigu = 750.0
omega = 100.0


def yield_stress(p):
    return sigu + (sig0 - sigu) * np.exp(-p * omega)


material = ElastoPlasticIsotropicHardening(elastic_model, yield_stress)
N = 10
Exx = np.concatenate(
    (
        np.linspace(0, 2e-2, N + 1),
        np.linspace(2e-2, 1e-2, N + 1)[1:],
        np.linspace(1e-2, 3e-2, N + 1)[1:],
    )
)
uniaxial_test_2D(material, Exx, N=1, order=1)
