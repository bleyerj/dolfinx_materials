import numpy as np
from dolfinx_materials.materials.mfront.mfront_material import MFrontNonlinearMaterial
from uniaxial_test import uniaxial_test_2D
import ufl
from dolfinx_materials.quadrature_map import QuadratureMap
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh
import matplotlib.pyplot as plt

from dolfinx_materials.solvers import CustomNewton


domain = mesh.create_unit_square(MPI.COMM_WORLD, 2 , 2, mesh.CellType.quadrilateral)
V = fem.VectorFunctionSpace(domain, ("CG", 2))
deg_quad = 2

E = 100e3
nu = 0.0
sig0 = 500.0
alpha = 2e-3 * E / sig0
n = 100.0

material = MFrontNonlinearMaterial(
    "dolfinx_materials/materials/mfront/src/libBehaviour.so", "RambergOsgoodNonLinearElasticity",
    material_properties={"YoungModulus": E,
                         "PoissonRatio": nu,
                         "YieldStrength": sig0,
                         "alpha": alpha,
                         "n": n})

N = 10
Exx = np.linspace(0, 2e-2, N + 1)
# Exx = np.concatenate(
#     (
#         np.linspace(0, 2e-2, N + 1),
#         np.linspace(2e-2, 1e-2, N + 1)[1:],
#         np.linspace(1e-2, 3e-2, N + 1)[1:],
#     )
# )

uniaxial_test_2D(material, Exx, 2, 1)
