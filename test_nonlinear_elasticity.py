import numpy as np
from dolfinx_materials.materials.python import RambergOsgood
from uniaxial_test import uniaxial_test_2D


mat_props = {
    "E": 100e3,
    "nu": 0.3,
    "sig0": 500.0,
    "alpha": 2e-3 * 100e3 / 500.0,
    "n": 100.0,
}
material = RambergOsgood(**mat_props)

N = 10
Exx = np.linspace(0, 2e-2, N + 1)
# Exx = np.concatenate(
#     (
#         np.linspace(0, 2e-2, N + 1),
#         np.linspace(2e-2, 1e-2, N + 1)[1:],
#         np.linspace(1e-2, 3e-2, N + 1)[1:],
#     )
# )

uniaxial_test_2D(material, Exx, 1)
