import pathlib
import numpy as np
import sys
import pytest

from dolfinx_materials.material.mfront import MFrontMaterial

path = pathlib.Path(__file__).parent.absolute()

sys.path.append(str(path.parent.absolute()))
from uniaxial_tension import uniaxial_tension_2D


@pytest.mark.parametrize("mesh_size", [1, 2, 4])
def test_mfront_elastoplasticity(mesh_size):
    sig0 = 250.0
    material = MFrontMaterial(
        path / "src/libBehaviour.so",
        "IsotropicLinearHardeningPlasticity",
        material_properties={
            "YoungModulus": 70e3,
            "PoissonRatio": 0.3,
            "HardeningSlope": 1e-6,
            "YieldStrength": sig0,
        },
    )

    N = 50
    Exx = np.linspace(0, 2e-2, N + 1)
    Stress = uniaxial_tension_2D(material, Exx, mesh_size)
    assert np.allclose(
        Stress[-1, :3], 2 / np.sqrt(3) * np.array([sig0, 0, sig0 / 2]), rtol=1e-2
    )


# mesh_size = 4
# test_mfront_elastoplasticity(mesh_size)
