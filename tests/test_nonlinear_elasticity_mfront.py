import numpy as np
import os

from dolfinx_materials.material.mfront import MFrontMaterial


os.path.join(os.path.abspath(__file__))
from uniaxial_tension import uniaxial_tension_2D

E = 100e3
nu = 0.3
sig0 = 500.0
alpha = 2e-3 * E / sig0
n = 100.0


def test_mfront_RambergOsgood():
    material = MFrontMaterial(
        "dolfinx_materials/mfront_materials/src/libBehaviour.so",
        "RambergOsgoodNonLinearElasticity",
        material_properties={
            "YoungModulus": E,
            "PoissonRatio": nu,
            "YieldStrength": sig0,
            "alpha": alpha,
            "n": n,
        },
    )

    N = 21
    Exx = np.linspace(0, 1e-2, N + 1)
    Stress = uniaxial_tension_2D(material, Exx, 1, 1)
    Stress[:, 3:] /= np.sqrt(2)
    np.savetxt(
        "tests/RambergOsgood_dolfinx_mfront.csv",
        np.concatenate((Exx[:, np.newaxis], Stress), axis=1),
        header="EXX SXX SYY SZZ SXY SXZ SYZ",
        delimiter=",",
    )


def test_against_Mtest():
    res_mtest = np.loadtxt("tests/mtest/RambergOsgood.csv", skiprows=1, delimiter=",")
    # test_mfront_RambergOsgood()
    res_dolfinx = np.loadtxt(
        "tests/RambergOsgood_dolfinx_mfront.csv", skiprows=1, delimiter=","
    )
    S_mtest = res_mtest[:, 7:10]
    S_dolfinx = res_dolfinx[:, 1:4]
    assert np.allclose(S_mtest, S_dolfinx, rtol=1e-4)
