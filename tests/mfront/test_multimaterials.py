#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole Nationale des Ponts et Chaussées, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   20/12/2023
"""
import pathlib
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.mfront import MFrontMaterial
from dolfinx_materials.utils import (
    symmetric_tensor_to_vector,
)


def test_multimaterials():
    path = pathlib.Path(__file__).parent.absolute()

    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        2,
        2,
        cell_type=mesh.CellType.quadrilateral,
    )
    gdim = domain.geometry.dim

    def left(x):
        return x[0] <= 0.5

    def right(x):
        return x[0] >= 0.5

    num_entities_local = domain.topology.index_map(gdim).size_local
    cells_l = mesh.locate_entities(domain, gdim, left)
    cells_l = cells_l[cells_l < num_entities_local]  # remove ghost entities

    cells_r = mesh.locate_entities(domain, gdim, right)
    cells_r = cells_r[cells_r < num_entities_local]  # remove ghost entities

    marked_entities = np.hstack((cells_l, cells_r))
    marked_values = np.hstack((np.full_like(cells_l, 1), np.full_like(cells_r, 2)))

    sorted_entities = np.argsort(marked_entities)
    subdomains = mesh.meshtags(
        domain, gdim, marked_entities[sorted_entities], marked_values[sorted_entities]
    )

    V = fem.functionspace(domain, ("Q", 1, (gdim,)))

    u = fem.Function(V, name="Displacement")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    def eps(u):
        return symmetric_tensor_to_vector(ufl.sym(ufl.grad(u)))

    x = ufl.SpatialCoordinate(domain)
    exx = fem.Constant(domain, 1e-2)
    E_macro = ufl.as_matrix([[exx, 0], [0, 0]])
    u_exp = fem.Expression(ufl.dot(E_macro, x), V.element.interpolation_points)

    domain.topology.create_connectivity(gdim - 1, gdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, gdim - 1, boundary_facets)
    uD = fem.Function(V)
    bcs = [fem.dirichletbc(uD, boundary_dofs)]

    mat_prop = {
        "YoungModulus": 70e3,
        "PoissonRatio": 0.3,
        "HardeningSlope": 5e3,
        "YieldStrength": 250.0,
    }

    material, material_l, material_r = [
        MFrontMaterial(
            path / "src/libBehaviour.so",
            "IsotropicLinearHardeningPlasticity",
            hypothesis="plane_strain",
            material_properties=mat_prop,
        )
        for i in range(3)
    ]

    qmap = QuadratureMap(domain, 2, material)
    qmap.register_gradient("Strain", eps(u))
    sig = qmap.fluxes["Stress"]
    qmap.initialize_state()

    qmap_l = QuadratureMap(domain, 2, material_l, cells=cells_l)
    qmap_l.register_gradient("Strain", eps(u))
    sig_l = qmap_l.fluxes["Stress"]
    qmap_l.initialize_state()

    qmap_r = QuadratureMap(domain, 2, material_r, cells=cells_r)
    qmap_r.register_gradient("Strain", eps(u))
    sig_r = qmap_r.fluxes["Stress"]
    qmap_r.initialize_state()

    newton = NewtonSolver(MPI.COMM_WORLD)
    newton.rtol = 1e-6
    newton.max_it = 10

    Exx = np.linspace(0, 1e-2, 10)

    # mono-materiel problem
    Res = ufl.dot(sig, eps(v)) * qmap.dx
    Jac = qmap.derivative(Res, u, du)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_atol": 1e-10,
        "snes_rtol": 1e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = NonlinearMaterialProblem(
        qmap,
        Res,
        u,
        bcs=bcs,
        J=Jac,
        petsc_options_prefix="test_mono",
        petsc_options=petsc_options,
    )
    Sig = np.zeros((len(Exx), len(sig.x.array)))
    for i, exxi in enumerate(Exx[1:]):
        exx.value = exxi
        uD.interpolate(u_exp)

        problem.solve()

        assert problem.solver.getConvergedReason()

        Sig[i + 1, :] = sig.x.array

    # multi-material problem
    Res_l = ufl.dot(sig_l, eps(v)) * qmap_l.dx
    Res_r = ufl.dot(sig_r, eps(v)) * qmap_r.dx
    Jac_l = qmap_l.derivative(Res_l, u, du)
    Jac_r = qmap_r.derivative(Res_r, u, du)

    problem = NonlinearMaterialProblem(
        [qmap_l, qmap_r],
        Res_l + Res_r,
        u,
        bcs=bcs,
        J=Jac_l + Jac_r,
        petsc_options_prefix="test_multi",
        petsc_options=petsc_options,
    )

    for i, exxi in enumerate(Exx[1:]):
        exx.value = exxi
        uD.interpolate(u_exp)

        problem.solve()

        assert problem.solver.getConvergedReason()

        assert np.allclose(sig_l.x.array * sig_r.x.array, np.zeros_like(sig.x.array))
        assert np.allclose(Sig[i + 1, :], sig_l.x.array + sig_r.x.array)
