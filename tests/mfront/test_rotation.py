#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   20/12/2023
"""
import pathlib
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx_materials.utils import (
    symmetric_tensor_to_vector,
)

path = pathlib.Path(__file__).parent.absolute()

domain = mesh.create_unit_cube(
    MPI.COMM_WORLD,
    1,
    1,
    1,
    cell_type=mesh.CellType.hexahedron,
)
gdim = domain.topology.dim


V = fem.functionspace(domain, ("Q", 1, (gdim,)))


u = fem.Function(V, name="Displacement")


def eps(u):
    return symmetric_tensor_to_vector(ufl.sym(ufl.grad(u)))


x = ufl.SpatialCoordinate(domain)
exx = fem.Constant(domain, 1e-4)
E_macro = ufl.as_matrix([[exx, 0, 0], [0, 0, 0], [0, 0, 0]])
u_exp = fem.Expression(ufl.dot(E_macro, x), V.element.interpolation_points())


def rotation_symmetry(material, phi, isotropic):
    qmap = QuadratureMap(domain, 2, material)
    qmap.register_gradient("Strain", eps(u))
    sig = qmap.fluxes["Stress"]
    qmap.initialize_state()

    u.interpolate(u_exp)
    qmap.update()

    Sig = np.copy(sig.x.array)
    # # test that the material is isotropic
    for phi_i in [np.pi / 3, np.pi / 4, np.pi / 2]:
        phi.value = phi_i
        u.x.petsc_vec.set(0.0)

        qmap = QuadratureMap(domain, 2, material)
        qmap.register_gradient("Strain", eps(u))
        sig = qmap.fluxes["Stress"]
        qmap.initialize_state()

        u.interpolate(u_exp)

        qmap.update()
        assert np.allclose(Sig, sig.x.array) == isotropic


def test_rotation_isotropy():
    phi = fem.Constant(domain, 0.0)
    R = ufl.as_matrix(
        [
            [ufl.cos(phi), ufl.sin(phi), 0],
            [-ufl.sin(phi), ufl.cos(phi), 0],
            [0, 0, 1],
        ]
    )

    material = MFrontMaterial(
        path / "src/libBehaviour.so",
        "MericCailletaudSingleCrystalViscoPlasticity",
        rotation_matrix=R,
        material_properties={"YoungModulus1": 208000.0},
    )
    rotation_symmetry(material, phi, True)

    material = MFrontMaterial(
        path / "src/libBehaviour.so",
        "MericCailletaudSingleCrystalViscoPlasticity",
        rotation_matrix=R,
        material_properties={"YoungModulus1": 100000.0},
    )
    rotation_symmetry(material, phi, False)
