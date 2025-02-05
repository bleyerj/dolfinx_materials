import pathlib
import numpy as np
import sys
import pytest
import ufl
from dolfinx_materials.material.mfront import MFrontMaterial

path = pathlib.Path(__file__).parent.absolute()

sys.path.append(str(path.parent.absolute()))
from uniaxial_tension import uniaxial_tension_2D


@pytest.mark.parametrize("mesh_size", [1, 2, 4])
def test_mfront_elastoplasticity(mesh_size):
    sig0 = 250.0
    material = MFrontMaterial(
        path / "src/libBehaviour.so",
        "Plasticity",
        material_properties={
            "YoungModulus": 70e3,
            "PoissonRatio": 0.3,
            "HardeningSlope": 1e-6,
            "YieldStrength": sig0,
        },
    )

    N = 50
    Exx = np.linspace(0, 2e-2, N + 1)
    Stress = uniaxial_tension_2D(material, Exx, N=mesh_size)
    assert np.allclose(
        Stress[-1, :3], 2 / np.sqrt(3) * np.array([sig0, 0, sig0 / 2]), rtol=1e-2
    )


def test_mfront_single_cristal():
    sig0 = 250.0
    material = MFrontMaterial(
        path / "src/libBehaviour.so",
        "MericCailletaudSingleCrystalViscoPlasticity",
        material_properties={"YoungModulus1": 208000.0},
    )
    material.dt = 1e-1

    N = 50
    Exx = np.linspace(0, 5e-3, N + 1)
    angles = [0.0, np.pi / 4, np.pi / 3, np.pi / 2]
    Stresses = []
    for angle in angles:
        Stresses.append(uniaxial_tension_2D(material, Exx, N=1, angle=angle))
    for i in range(4):
        # First increment is in elastic regime, all stresses should be the same
        assert np.allclose(Stresses[i][1, :], Stresses[(i + 1) % 4][1, :])

    # Check that final plastic state is the same for 0deg and 90deg
    assert np.allclose(Stresses[0][-1, :], Stresses[3][-1, :])
    # Check that final plastic state is not the same for 45deg and 60deg
    assert not (np.allclose(Stresses[0][-1, :], Stresses[1][-1, :]))
    assert not (np.allclose(Stresses[0][-1, :], Stresses[2][-1, :]))
