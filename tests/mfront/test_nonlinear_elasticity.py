import numpy as np
import os
import sys

from dolfinx_materials.material.mfront import MFrontMaterial


current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from uniaxial_tension import uniaxial_tension_2D

E = 100e3
nu = 0.3
sig0 = 500.0
alpha = 2e-3 * E / sig0
n = 100.0


def test_mfront_RambergOsgood():
    material = MFrontMaterial(
        "src/libBehaviour.so",
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
        "RambergOsgood_dolfinx_mfront.csv",
        np.concatenate((Exx[:, np.newaxis], Stress), axis=1),
        header="EXX SXX SYY SZZ SXY SXZ SYZ",
        delimiter=",",
    )


def test_against_Mtest():
    res_mtest = np.loadtxt("mtest/RambergOsgood.csv", skiprows=1, delimiter=",")
    res_dolfinx = np.loadtxt(
        "RambergOsgood_dolfinx_mfront.csv", skiprows=1, delimiter=","
    )
    S_mtest = res_mtest[:, 7:10]
    S_dolfinx = res_dolfinx[:, 1:4]
    assert np.allclose(S_mtest, S_dolfinx, rtol=1e-4)
