import pathlib
import numpy as np
import sys
import pytest
import ufl
from dolfinx_materials.mfront import MFrontMaterial

path = pathlib.Path(__file__).parent.absolute()

sys.path.append(str(path.parent.absolute()))


def test_mfront_finite_strain_options():
    material = MFrontMaterial(
        path / "src/libBehaviour.so",
        "LogarithmicStrainPlasticity",
    )
    assert material.behaviour.getBehaviourType() == "StandardFiniteStrainBehaviour"
    assert material.behaviour.getKinematic() == "F_CAUCHY"
    assert material.gradient_names == ["DeformationGradient"]
    assert material.flux_names == ["FirstPiolaKirchhoffStress"]
    assert material.tangent_block_names == [
        ("FirstPiolaKirchhoffStress", "DeformationGradient")
    ]

    material = MFrontMaterial(
        path / "src/libBehaviour.so",
        "LogarithmicStrainPlasticity",
        finite_strain_options={"stress": "PK2", "tangent": "DS_DEGL"},
    )
    assert material.behaviour.getBehaviourType() == "StandardFiniteStrainBehaviour"
    assert material.behaviour.getKinematic() == "F_CAUCHY"
    assert material.gradient_names == ["DeformationGradient"]
    assert material.flux_names == ["SecondPiolaKirchhoffStress"]
    assert material.tangent_block_names == [
        ("SecondPiolaKirchhoffStress", "GreenLagrangeStrain")
    ]
