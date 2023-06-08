import numpy as np
from dolfinx_materials.materials.python import (
    NielsenPlate,
    PlaneStressLinearElasticIsotropic,
)
from dolfinx import fem, mesh, io, nls
from dolfinx.cpp.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem

Nx = 10
cell_type = mesh.CellType.quadrilateral
domain = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Nx, cell_type)

E, nu = 30e3, 0.2
elastic_model = PlaneStressLinearElasticIsotropic()
elastic_model.E = E
elastic_model.nu = nu
elastic_model.C = elastic_model.compute_C(E, nu)

thick = 1e-1
sig0 = 30.0
mp0 = sig0 * thick**2
beta = 10.0
F = fem.Constant(domain, 5 / 6.0 * E / 2 / (1 + nu) * thick)
yield_parameters = {"mxp": mp0, "myp": beta * mp0, "mxm": mp0, "mym": beta * mp0}
material = NielsenPlate(elastic_model, thick, yield_parameters)


deg = 2
We = ufl.FiniteElement("Lagrange", domain.ufl_cell(), deg)
Te = ufl.VectorElement("Lagrange", domain.ufl_cell(), deg)
V = fem.FunctionSpace(domain, ufl.MixedElement([We, Te]))

deg_quad = 1


def border(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1.0)),
        np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1.0)),
    )


uD = fem.Function(V)
facets = mesh.locate_entities_boundary(domain, 1, border)
bcs = [fem.dirichletbc(uD, fem.locate_dofs_topological(V, 1, facets))]

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V)


def shear_strain(u):
    (w, theta) = ufl.split(u)
    return theta - ufl.grad(w)


def curvature(u):
    w, theta = ufl.split(u)
    return ufl.as_vector(
        [
            theta[0].dx(0),
            theta[1].dx(1),
            1 / np.sqrt(2) * (theta[1].dx(0) + theta[0].dx(1)),
        ]
    )


dx = ufl.Measure("dx", domain=domain)
qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient("χ", curvature(u))
f = fem.Constant(domain, 0.0)
print(qmap.material.tangent_blocks)
# Res = (
#     ufl.dot(qmap.fluxes["M"], curvature(v)) * qmap.dx
#     + (F * ufl.dot(shear_strain(u), shear_strain(v)) - f * v[0]) * dx
# )
Res = (
    ufl.dot(curvature(u), curvature(v)) * qmap.dx
    + (F * ufl.dot(shear_strain(u), shear_strain(v)) - f * v[0]) * dx
)
Jac = ufl.derivative(Res, u, du)

# problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)
problem = fem.petsc.NonlinearProblem(Res, u, bcs, Jac)
newton = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
# newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-4
newton.convergence_criterion = "incremental"
newton.report = True


xdmf = io.XDMFFile(domain.comm, "plates.xdmf", "w")
xdmf.write_mesh(domain)

f_list = np.linspace(0, 10.0, 10)
u_max = np.zeros_like(f_list)
for i, fi in enumerate(f_list[1:]):
    f.value = fi

    # converged, it = problem.solve(newton)
    converged, it = newton.solve(u)
    print(converged)

    w = u.sub(0).collapse()
    w.name = "Deflection"
    xdmf.write_function(w, i)

    u_max[i + 1] = -min(u.vector.array)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(u_max, f_list, "-o")
plt.xlabel(r"Deflection")
plt.ylabel(r"Load")
plt.savefig(f"{material.name}_deflection.pdf")
