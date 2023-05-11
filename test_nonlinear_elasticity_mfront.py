import numpy as np
from dolfinx_materials.materials.mfront.mfront_material import MFrontNonlinearMaterial
from uniaxial_test import uniaxial_test_2D

E = 100e3
nu = 0.0
sig0 = 500.0
alpha = 2e-3 * E / sig0
n = 100.0

material = MFrontNonlinearMaterial(
    "materials/mfront/src/libBehaviour.so", "RambergOsgoodNonLinearElasticity"
)
variables = {"eps": 6, "sig": 6}

N = 10
Exx = np.linspace(0, 2e-2, N + 1)
# Exx = np.concatenate(
#     (
#         np.linspace(0, 2e-2, N + 1),
#         np.linspace(2e-2, 1e-2, N + 1)[1:],
#         np.linspace(1e-2, 3e-2, N + 1)[1:],
#     )
# )

uniaxial_test_2D(material, variables, Exx)
