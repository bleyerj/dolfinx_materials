#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole des Ponts ParisTech, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   20/12/2023
"""
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx_materials.utils import (
    symmetric_tensor_to_vector,
)


domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    1,
    1,
    cell_type=mesh.CellType.quadrilateral,
)
gdim = domain.topology.dim


V = fem.functionspace(domain, ("Q", 1, (gdim,)))


u = fem.Function(V, name="Displacement")


def eps(u):
    return symmetric_tensor_to_vector(ufl.sym(ufl.grad(u)))


x = ufl.SpatialCoordinate(domain)
exx = fem.Constant(domain, 1e-2)
E_macro = ufl.as_matrix([[exx, 0], [0, 0]])
u_exp = fem.Expression(ufl.dot(E_macro, x), V.element.interpolation_points())

mat_prop = {
    "YoungModulus": 70e3,
    "PoissonRatio": 0.3,
    "HardeningSlope": 5e3,
    "YieldStrength": 250.0,
}
material = MFrontMaterial(
    "src/libBehaviour.so",
    "IsotropicLinearHardeningPlasticity",
    hypothesis="plane_strain",
    material_properties=mat_prop,
)

qmap = QuadratureMap(domain, 2, material)
qmap.register_gradient("Strain", eps(u))
sig = qmap.fluxes["Stress"]
p = qmap.internal_state_variables["EquivalentPlasticStrain"]

qmap.initialize_state()

u.interpolate(u_exp)

p.vector.array[:] = 1e-4
# without value
qmap.update_initial_state("EquivalentPlasticStrain")
assert np.allclose(p.vector.array, 1e-4)

# with value
qmap.update_initial_state("EquivalentPlasticStrain", 1e-3)
assert np.allclose(p.vector.array, 1e-3)

# with fem.Constant
p0 = fem.Constant(domain, 2e-3)
qmap.update_initial_state("EquivalentPlasticStrain", p0)
assert np.allclose(p.vector.array, 2e-3)

# with fem.Function
V0 = fem.FunctionSpace(domain, ("DG", 0))
p0 = fem.Function(V0)
p0.vector.set(3e-3)
qmap.update_initial_state("EquivalentPlasticStrain", p0)
assert np.allclose(p.vector.array, 3e-3)

# vectorial case
# with fem.Constant
s0 = np.array([1.0, 2.0, 3.0, 4.0])
sig0 = fem.Constant(domain, s0)
qmap.update_initial_state("Stress", sig0)
assert np.allclose(sig.vector.array, np.tile(s0, 4))

# with numpy array
qmap.update_initial_state("Stress", 2 * s0)
assert np.allclose(sig.vector.array, np.tile(2 * s0, 4))
